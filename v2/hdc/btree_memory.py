"""
HDC BTree Memory Layer
Implémente une mémoire associative structurée en BTree pour un accès O(log N).
Chaque nœud est une clé binaire (LSH) menant à un Hypervecteur ou un sous-arbre.
"""

import numpy as np

class BTreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.keys = []     # Clés de hash (int)
        self.values = []   # Valeurs (Hypervecteurs ou listes de phrases)
        self.children = [] # Enfants (BTreeNode)

class BinaryHDCBTree:
    def __init__(self, t=10):
        """
        t : degré minimum du BTree.
        Un nœud a au moins t-1 clés et au plus 2t-1 clés.
        """
        self.root = BTreeNode(True)
        self.t = t

    def insert(self, key, value):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            new_root = BTreeNode(False)
            self.root = new_root
            new_root.children.insert(0, root)
            self._split_child(new_root, 0)
            self._insert_non_full(new_root, key, value)
        else:
            self._insert_non_full(root, key, value)

    def _insert_non_full(self, x, k, v):
        i = len(x.keys) - 1
        if x.is_leaf:
            x.keys.append(0)
            x.values.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i+1] = x.keys[i]
                x.values[i+1] = x.values[i]
                i -= 1
            x.keys[i+1] = k
            x.values[i+1] = v
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self._split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self._insert_non_full(x.children[i], k, v)

    def _split_child(self, x, i):
        t = self.t
        y = x.children[i]
        z = BTreeNode(y.is_leaf)
        x.children.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        x.values.insert(i, y.values[t - 1])
        
        z.keys = y.keys[t : (2 * t) - 1]
        z.values = y.values[t : (2 * t) - 1]
        y.keys = y.keys[0 : t - 1]
        y.values = y.values[0 : t - 1]
        
        if not y.is_leaf:
            z.children = y.children[t : 2 * t]
            y.children = y.children[0 : t]

    def search(self, k, x=None):
        if x is None:
            x = self.root
        i = 0
        while i < len(x.keys) and k > x.keys[i]:
            i += 1
        if i < len(x.keys) and k == x.keys[i]:
            return x.values[i]
        elif x.is_leaf:
            return None
        else:
            return self.search(k, x.children[i])
