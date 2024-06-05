import numpy as np


class QuinticHermiteSplines:
    def __init__(self):
        self.start_point_x = 0
        self.start_point_dx = 0
        self.start_point_ddx = 0
        self.end_point_x = 0
        self.end_point_dx = 0
        self.end_point_ddx = 0
        self.c0 = 0
        self.c1 = 0
        self.c2 = 0
        self.c3 = 0
        self.c4 = 0
        self.c5 = 0

    def set_start_point(self, x, dx=0, ddx=0):
        self.start_point_x = x
        self.start_point_dx = dx
        self.start_point_ddx = ddx

    def set_end_point(self, x, dx=0, ddx=0):
        self.end_point_x = x
        self.end_point_dx = dx
        self.end_point_ddx = ddx

    def solve(self):
        self.c0 = self.start_point_x
        self.c1 = self.start_point_dx
        self.c2 = self.start_point_ddx / 2
        A = np.array([[-10, -6, -3 / 2, 10, -4, 1 / 2],
                      [15, 8, 3 / 2, -15, 7, -1],
                      [-6, -3, -1 / 2, 6, -3, 1 / 2]])
        x = np.array(
            [self.start_point_x, self.start_point_dx, self.start_point_ddx, self.end_point_x, self.end_point_dx,
             self.end_point_ddx])
        b = np.dot(A, x)
        self.c3 = b[0]
        self.c4 = b[1]
        self.c5 = b[2]

    def get_point(self, t):
        return self.c0 + self.c1 * t + self.c2 * t ** 2 + self.c3 * t ** 3 + self.c4 * t ** 4 + self.c5 * t ** 5


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')
    x_spline = QuinticHermiteSplines()
    y_spline = QuinticHermiteSplines()
    x_spline.set_start_point(0, 0, 0)
    x_spline.set_end_point(0, 0, 0)
    print(x_spline.c0)
    print(x_spline.c1)
    print(x_spline.c2)
    y_spline.set_start_point(1.6, 0, 0)
    y_spline.set_end_point(1.6, 0, 0)
    x_spline.solve()
    y_spline.solve()
    x = []
    y = []
    for i in range(100):
        x.append(x_spline.get_point(i / 100))
        y.append(y_spline.get_point(i / 100))
    plt.plot(y)
    plt.show()
