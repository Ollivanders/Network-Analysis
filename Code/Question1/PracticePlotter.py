import matplotlib.pyplot as plt

plt.figure()

colour = (1, 0, 0, 1)

x_data = []
y_data = []

p = 0.5
q = 0.0

for value in range(1, 10000):
    print(str(value))
    q = value*0.25/10000
    p = 0.5 - q
    plt.plot(q, p, marker='.', color=colour)  # linestyle='None'
    amount_of_green = (1 / 10000) * value
    colour = (1, amount_of_green, 0)

plt.xlabel('Value of q')
plt.ylabel('Value of p')

plt.tight_layout()
#plt.gca().invert_xaxis()
plt.savefig('./Legend.png')

