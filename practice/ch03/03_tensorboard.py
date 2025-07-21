import math
from torch.utils.tensorboard.writer import SummaryWriter


if __name__ == "__main__":
    # 텐서보드 작성자 생성
    writer = SummaryWriter()

    # 함수 딕셔너리 생성
    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    # 각도 범위 내에서 함수 값 계산 및 텐서보드에 기록
    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)

    writer.close()
