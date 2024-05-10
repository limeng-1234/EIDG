import numpy as np
import time
import xgboost as xgb
from typing import Callable, Union, List, Tuple
import numpy as np

### 辅助函数，用于处理多输出问题
def model_modity(model, input, i, output_ndim):
    if output_ndim == 1:
        return model(input)[:]
    return model(input)[:, i]

class EIDG:
    def __init__(self, x_baselines: np.ndarray, model: Callable, steps: int = 50, h: float = 0.1, m: int = 10, pos: bool = True):
        """
        初始化函数，用于设置类的参数。
        参数：
        - x_baselines: numpy array，背景样本，可以使用多个基值，形状为 (n, d)，其中 n 为背景样本数量，d 为特征向量的维度。
        - model: function。
        - steps: int，积分步数，默认为 50。
        - h: float，微小增量，用于计算导数，默认为 0.1。
        - m: int，正定基值数量，默认为10
        - pos: bool，是否使用正定基值
        """
        self.x_baselines = x_baselines
        self.model = model
        self.steps = steps
        self.h = h
        self.m = m
        self.pos = pos
        self.baselines_number = len(self.x_baselines)

        self.baselines_selection()
        self.base_value = np.mean(self.model(self.x_baselines), axis=0)

    def baselines_selection(self):
        if self.baselines_number < self.m:
            raise ValueError('the number of x_baselines must be greater than or equal to m')
        if self.pos:
            output = self.model(self.x_baselines)
            if output.ndim == 1:
                min_indices = np.argsort(output)[:self.m]
                self.x_baselines = self.x_baselines[min_indices]
                self.base_value = np.mean(self.model(self.x_baselines))
            elif output.ndim == 2:
                max_values = [max(row) for row in output]
                min_indices = np.argsort(max_values)[:self.m]
                self.x_baselines = self.x_baselines[min_indices]
                self.base_value = np.mean(self.model(self.x_baselines),axis=0)
            else:
                raise ValueError("Unsupported output shape")
        else:
            self.x_baselines = self.x_baselines[np.random.choice(self.x_baselines.shape[0], self.m, replace=False)]
        self.baselines_number = len(self.x_baselines)

    def integrated_gradients(self, sample):
        """
        - x_baselines: numpy array，计算样本，形状为 (n, d)，其中n样本数量，d为特征向量的维度。
        """
        start_time = time.time()

        alphas = np.linspace(0, 1, self.steps)
        weights = np.zeros((sample.shape[1], len(self.x_baselines) * self.steps))

        integral_paths_list = []
        for x in sample:
            for x_baseline in self.x_baselines:
                paths = x_baseline + alphas[:, np.newaxis] * (x - x_baseline)
                integral_paths_list.append(paths)
        integral_paths = np.array(integral_paths_list)
        integral_paths = np.reshape(integral_paths, (-1, integral_paths.shape[2]))

        samples_number = len(sample)
        weights = np.zeros((sample.shape[1], samples_number * self.baselines_number * self.steps))

        self.integrated_gradient_list = []
        output_ndim = self.model(sample).ndim

        for output_number in range(output_ndim):
            integrated_gradient_values = np.zeros_like(sample)
            for i in range(sample.shape[1]):
                x_prime_plus_h = integral_paths.copy()
                x_prime_plus_h[:, i] += self.h
                x_prime_minus_h = integral_paths.copy()
                x_prime_minus_h[:, i] -= self.h

                delta_F = (model_modity(self.model, x_prime_plus_h, output_number, output_ndim) - model_modity(self.model, x_prime_minus_h, output_number, output_ndim))
                weights[i] = delta_F / 2 / self.h
            for i, x in enumerate(sample):
                for j, x_baseline in enumerate(self.x_baselines):
                    integrated_gradient_values[i, :] += (x - x_baseline) * np.sum(weights[:,
                                                                            i * self.baselines_number * self.steps + j * self.steps: i * self.baselines_number * self.steps + (
                                                                                        j + 1) * self.steps],
                                                                            axis=1) / self.steps / self.baselines_number
            self.integrated_gradient_list.append(integrated_gradient_values)
        end_time = time.time()
        self.elapsed_time = end_time - start_time

        print("EIDG运行时间：", self.elapsed_time, "秒")

        return self.integrated_gradient_list
