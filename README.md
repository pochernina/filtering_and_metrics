## Фильтрация и метрики
### Задание №2 в рамках спецсеминара «Математические методы обработки изображений»

Программа реализует следующие алгоритмы фильтрации изображения и метрики:
* медианная фильтрация изображения
* фильтр Гаусса
* билатеральная фильтрация
* метрика MSE
* метрика PSNR
* метрика SSIM

### Запуск из командной строки:
В случае метрик:

``` bash
python main.py metric input_file_1 input_file_2
```
Список метрик:
* `mse`
* `psnr`
* `ssim`

Размеры изображений input_file_1 и input_file_2 должны совпадать.

---

В случае фильтраций:
``` bash
python main.py filter parameters input_file output_file
```
Список фильтров:
* `median` `rad` — медианная фильтрация с окном размера (2rad+1) × (2rad+1)
* `gauss` `sigma_d` — фильтр Гаусса с параметром <img src="https://render.githubusercontent.com/render/math?math=\sigma_d">
* `bilateral` `sigma_d sigma_r` — билатеральная фильтрация с параметрами <img src="https://render.githubusercontent.com/render/math?math=\sigma_d"> и <img src="https://render.githubusercontent.com/render/math?math=\sigma_r">

Значение rad — целое положительное, значения sigma_d и sigma_r — вещественные положительные.

---

В файле `main_with_cv2.py` находится реализация с использованием готовых функций из библиотек scikit-image, opencv-python. В файле `main.py` находятся собственные реализации.