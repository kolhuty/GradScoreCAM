# GradScoreCAM

Реализация гибридного метода интерпретации CNN.
Метод ускоряет Score-CAM за счет отбора наиболее важных карт активации через градиенты. 

Метод GradScoreCAM реализован через библиотеку `grad-cam` в `method.py`.  
В `gibrid-metrics.ipynb` проведено сравнение 4 методов на датасете ImageNet по метрикам Insection AUC, Delection AUC и времени исполнения.
