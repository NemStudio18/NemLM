import multiprocessing as mp
import time
import os
from hdc.memory import AssociativeMemory
from hdc.representation import encode_context

def training_worker(task_queue, db_path, orders, dim):
    """
    Worker spécialisé dans certains ordres de n-grammes.
    Ouvre sa propre connexion SQLite.
    """
    # Initialisation de la mémoire (nouvelle connexion)
    memory = AssociativeMemory(dim, db_path=db_path)
    
    # Filtrage des singletons : on stocke les hashs pour économiser la RAM
    seen_once = set()
    
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        context_tokens, target_token = task
        
        for n in orders:
            # Vérifier si on a assez de contexte pour cet ordre
            if len(context_tokens) >= n - 1:
                # Extraire le sous-contexte pour l'ordre n
                if n == 1:
                    sub_context = []
                else:
                    sub_context = context_tokens[-(n-1):]
                
                # Clé unique pour le filtrage (hash du n-gramme complet)
                ngram_id = hash((tuple(sub_context), target_token))
                
                if ngram_id in seen_once:
                    # DEUXIÈME FOIS : On encode et on enregistre en base
                    n_gram_hv = encode_context(sub_context, dim)
                    memory.learn_one_pass(n_gram_hv, target_token)
                else:
                    # PREMIÈRE FOIS : On garde en mémoire RAM
                    seen_once.add(ngram_id)
        
        # Commit périodique pour libérer le WAL
        if task_queue.qsize() == 0:
            memory.commit()

    memory.commit()
    memory.close()

class V3ParallelEngine:
    def __init__(self, dim, db_path, num_workers=3):
        self.dim = dim
        self.db_path = db_path
        self.num_workers = num_workers
        self.task_queue = mp.Queue(maxsize=1000)
        self.processes = []
        
        # Spécialisation des workers par ordre
        # Worker 0: Ordre 2 (Bigrams)
        # Worker 1: Ordre 3 (Trigrams)
        # Worker 2: Ordre 4 & 5 (Syntaxe)
        order_sets = [[2], [3], [4, 5]]
        
        for i in range(num_workers):
            p = mp.Process(
                target=training_worker, 
                args=(self.task_queue, db_path, order_sets[i], dim)
            )
            p.start()
            self.processes.append(p)

    def train_step(self, sentence):
        """
        Envoie les tâches de la phrase aux workers.
        """
        if len(sentence) < 2: return
        
        for i in range(1, len(sentence)):
            context = sentence[:i]
            target = sentence[i]
            # On envoie la tâche complète, les workers filtreront ce qui les concerne
            self.task_queue.put((context, target))

    def commit(self):
        # On ne peut pas forcer le commit des workers directement ici,
        # ils le font périodiquement ou à la fin.
        pass

    def stop(self):
        # Envoyer le signal de fin à tous les workers
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        for p in self.processes:
            p.join()

    def predict_next(self, context_tokens, top_k=5):
        """
        L'inférence reste séquentielle (car elle est rapide et utilise son propre moteur).
        On réutilise une instance temporaire pour la prédiction.
        """
        # Note: Pour le duel, on utilise une instance propre pour prédire
        # car les workers sont occupés à écrire.
        from hdc.v3_engine import V3Engine
        predictor = V3Engine(self.dim, db_path=self.db_path)
        preds = predictor.predict_next(context_tokens, top_k=top_k)
        return preds
