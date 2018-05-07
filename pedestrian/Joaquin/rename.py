"""Renombra los archivos de imagenes a numeros
   Empieza numerando en el valor de i y va incrementando en 1"""
import os
path = '/Users/busca/Desktop/temporal/train/pos'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.png'))
    i = i+1