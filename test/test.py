from bess.bess import *
import numpy as np


np.random.seed(123)
# print("Random number with seed 123 : " + str(random.random()))

n = 10
p = 5

# # data parameter lm
train_X = np.random.normal(0, 1, n * p).reshape((n, p))
train_y = np.random.normal(0, 1, n)
test_X = np.random.normal(0, 1, 10 * p).reshape((10, p))      # test_x
model = PdasLm(is_cv=True, K=3)

# data parameter glm
# train_X = np.random.normal(0, 1, n * p).reshape((n, p))
# train_y = np.random.randint(0, 2, n)
# test_X = np.random.normal(0, 1, 10 * p).reshape((10, p))      # test_x
# model = PdasLogistic(path_type="gs", is_cv=True, K=3)

# print(train_X)
# print(train_y)

#
#
# # model parameter
# algorithm_type = "PDAS"   # "PDAS", "GROUP"
# model_type = "lm"         # "lm", "logistic"
# path_type = "sequence"    # "sequnece", "gs"

# with open("train_X_seq_PDAS_lm.txt", "w") as f:
#     for i in range(n):
#         for j in range(p):
#             f.write(train_X[i][j])
#             f.write(",")



# alternative parameter
# sequence = [1, 2, 3]
# s_min = 1
# s_max = 10
# K_max = 20
# epsilon = 0.00001


model.fit(train_X, train_y)

print(model.beta_out)
print(model.train_loss_out)
print(model.ic_out)
print(model.predict(test_X))


# m = 10
# test_x = np.random.normal(0, 1, m * p).reshape((m, p))
# y_pre = model.predict(test_x)
