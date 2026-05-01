"""
HDC Binary Persistence - Format Custom (Option B)
Permet de sauvegarder/charger la mémoire associative en binaire compact.
Structure : [Header] + N * [Hash64 | BitsPacked]
"""

import struct
import numpy as np
import os

MAGIC = b"HDC3"
VERSION = 1

def save_memory(file_path, storage, dim):
    """
    Sauvegarde le dictionnaire de stockage au format binaire.
    On ne sauvegarde que les BUNDLES (bits packés) pour l'inférence.
    """
    # Calcul de la taille d'un HV packé
    packed_size = (dim + 7) // 8
    
    print(f"💾 Sauvegarde de {len(storage)} entrées dans {file_path}...")
    
    with open(file_path, "wb") as f:
        # Header (16 bytes)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<I", 0)) # Reserved
        
        for key, entry in storage.items():
            # Clé (Hash 64-bit)
            f.write(struct.pack("<Q", key))
            
            # Valeur (HV bits packés)
            # On récupère le bundle (vote majoritaire)
            if hasattr(entry, 'bundle_cache') and entry.bundle_cache is not None:
                bundle = entry.bundle_cache
            elif hasattr(entry, 'weighted_sum'):
                bundle = (entry.weighted_sum > 0).astype(np.uint8)
            else:
                bundle = entry # Cas où c'est déjà un HV
                
            packed = np.packbits(bundle)
            f.write(packed.tobytes())

def load_memory(file_path):
    """
    Charge une mémoire binaire et retourne un dictionnaire {hash: hv}.
    """
    if not os.path.exists(file_path):
        return None, 0
    
    storage = {}
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("Format de fichier invalide (Magic mismatch)")
            
        version = struct.unpack("<I", f.read(4))[0]
        dim = struct.unpack("<I", f.read(4))[0]
        f.read(4) # Reserved
        
        packed_size = (dim + 7) // 8
        
        # Lecture des entrées
        entry_size = 8 + packed_size
        while True:
            data = f.read(entry_size)
            if not data:
                break
            
            key = struct.unpack("<Q", data[:8])[0]
            packed_hv = np.frombuffer(data[8:], dtype=np.uint8)
            # Décompression des bits vers uint8 [0, 1]
            hv = np.unpackbits(packed_hv)[:dim]
            storage[key] = hv
            
    print(f"📂 Chargement de {len(storage)} entrées réussi (D={dim}).")
    return storage, dim
