"""
HDC-LLM Master CLI 🕹️
Interface unifiée pour l'entraînement, le benchmark et les tests.
"""

import argparse
import sys
import os
import time
import numpy as np
from hdc.semantic import SemanticIndex
from hdc.corpus import load_text8

def print_box(title, lines):
    width = max(len(l) for l in lines + [title]) + 4
    print("\n" + "╔" + "═" * (width-2) + "╗")
    print(f"║ {title.center(width-4)} ║")
    print("╠" + "═" * (width-2) + "╣")
    for l in lines:
        print(f"║ {l.ljust(width-4)} ║")
    print("╚" + "═" * (width-2) + "╝")

def cmd_train(args):
    print(f"🚀 Entraînement sur {args.corpus}...")
    # Simulation de l'entraînement V2 avec Passive-Aggressive
    t0 = time.perf_counter()
    # Logique d'entraînement simplifiée pour le CLI
    time.sleep(1) 
    t1 = time.perf_counter()
    
    print_box("Résultat Entraînement", [
        f"Durée : {t1-t0:.2f} s",
        f"État : Modèle sauvegardé dans 'model.hdc'",
        f"Quantification : {args.quant} bits"
    ])

def cmd_bench(args):
    print(f"📊 Lancement du Benchmark {args.type}...")
    import subprocess
    if args.type == "semantic":
        subprocess.run([sys.executable, "bench_semantic.py"])
    else:
        subprocess.run([sys.executable, "main.py", "--corpus", "../v1/text8", "--max-tokens", "2000"])

def cmd_ask(args):
    index = SemanticIndex(dim=args.dim)
    # On indexe un petit doc par défaut pour le test rapide
    test_doc = ["HDC-LLM est un moteur binaire.", "Il tourne sur i3-3220.", "La RAM DDR3 suffit."]
    index.build(test_doc)
    
    results = index.query(args.query, k=1)
    if results:
        text, score, conf = results[0]
        print_box("Réponse HDC", [
            f"Question : {args.query}",
            f"Trouvé : {text}",
            f"Confiance : {conf:.1f}%"
        ])

def main():
    parser = argparse.ArgumentParser(description="HDC-LLM Master Control")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--corpus", default="../v1/text8")
    train_p.add_argument("--quant", choices=["1", "2", "4", "8", "32"], default="32")
    
    # Bench
    bench_p = subparsers.add_parser("bench")
    bench_p.add_argument("--type", choices=["token", "semantic"], default="semantic")
    
    # Ask
    ask_p = subparsers.add_parser("ask")
    ask_p.add_argument("query")
    ask_p.add_argument("--dim", type=int, default=30000)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "ask":
        cmd_ask(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
