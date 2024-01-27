# Quicksort with Cuda
El proyecto se basa en el archivo **GPU-Quicksort-jea.pdf**, una idea para aprovechar al máximo el paralelismo, la idea es que cada thread realice el conteo de una parte del vector. Una vez ha finalizado el conteo de cada thread, el thread 0 se encarga de realizar la suma de todos los threads del bloque para saber el índice del bloque en el vector final y se realiza la escritura en el vector final. Finalmente, los únicos elementos que quedarán por insertar serán los pivotes.

La descripción más detallada del proyecto se encuentra en el documento **Documentación.pdf**

## Compilación 
En la carpeta scr se encuentran todos los archivos para realizar la compilación en el entorno boada. Los comandos son:
```
make
sbatch job.sh
```
