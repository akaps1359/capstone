import numpy as np

class LowPassFilter:
    """
    EAR(눈동자 비율) 같은 데이터의 급격한 변화를 부드럽게 만들기 위한 EMA(Exponential Moving Average) 필터입니다.
    alpha 값이 0~1 사이이며, 작을수록 더 부드럽게(느리게) 반응합니다.
    """
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_value = None

    def update(self, value):
        if self.prev_value is None:
            self.prev_value = value
        else:
            self.prev_value = self.alpha * value + (1 - self.alpha) * self.prev_value
        return self.prev_value


class KalmanFilter1D:
    """
    Head Pitch(고개 각도) 처럼 떨림이 심하고 물리적 움직임이 있는 데이터에 적합한 1차원 칼만 필터입니다.
    """
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2, estimation_error=1.0, initial_value=0.0):
        # Q: Process Noise (시스템의 내재적 노이즈, 작을수록 모델을 신뢰)
        self.Q = process_noise
        # R: Measurement Noise (측정값의 노이즈, 클수록 측정값을 덜 신뢰)
        self.R = measurement_noise
        # P: Estimation Error Covariance (초기 추정 오차)
        self.P = estimation_error
        # X: State Estimate (추정값)
        self.X = initial_value

    def update(self, measurement):
        # 1. Prediction Update (Time Update)
        # 이번 단계에서는 단순 모델이라 예측값이 이전 상태와 같다고 가정 (X = X)
        self.P = self.P + self.Q

        # 2. Measurement Update
        K = self.P / (self.P + self.R)  # Kalman Gain
        self.X = self.X + K * (measurement - self.X)
        self.P = (1 - K) * self.P
        
        return self.X
