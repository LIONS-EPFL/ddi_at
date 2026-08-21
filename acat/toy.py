# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.8.3 ('virt')
#     language: python
#     name: python3
# ---

# %%
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# %%
u = jnp.array([1.0, 0.0])
v = jnp.array([-1.0, -1.0])/jnp.sqrt(2.0)

# %%
@jax.jit
def f(x, y):
    return y*(x[0]**2 + (x[1]+1)**2) + (1-y)*(x[0]**2 + (x[1]-1)**2) - y*(1-y)

# %%
gradf = jax.jit(jax.grad(f, argnums=(0, 1)))

# %%
def inner_max(x, step_size, n_steps, key):
    y = jax.random.uniform(key)
    for i in range(n_steps):
        y = y + step_size*gradf(x, y)[1]
        y = jnp.clip(y, 0, 1)
    return y

# %%
x = jnp.array([1.0, 0*(-1.0 - jnp.sqrt(2))])
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
x_lst_p = [x]
f_lst_p = [jnp.maximum((x[0]**2 + (x[1]+1)**2), (x[0]**2 + (x[1]-1)**2))]
step = 0.1
for k in range(1, 50):
    y_star = inner_max(x, 1, 10, subkey)
    x = x - 0.1 * gradf(x, y_star)[0]/jnp.linalg.norm(gradf(x, y_star)[0])
    x_lst_p.append(x)
    f_lst_p.append(jnp.maximum((x[0]**2 + (x[1]+1)**2), (x[0]**2 + (x[1]-1)**2)))
    key, subkey = jax.random.split(key)

# %%
from jax_descentdir import SimplexMinJax

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
x = jnp.array([1.0, 0*(-1.0 - jnp.sqrt(2))])
x_lst = [x]
f_lst = [jnp.maximum((x[0]**2 + (x[1]+1)**2), (x[0]**2 + (x[1]-1)**2))]
step = 0.1
for k in range(1, 50):
    y_lst = []
    for _ in range(10):
        key, subkey = jax.random.split(key)
        y_lst.append(inner_max(x, 1, 10, subkey))
    M = jnp.vstack([gradf(x, y)[0] for y in y_lst])
    MMT = jnp.dot(M, M.T)
    #print(jnp.sum(MMT < 0.0))
    danskin = SimplexMinJax(MMT, 10)
    mixture = danskin.optimize()
    y_star = jnp.dot(mixture, M)
    x = x - 0.1 * y_star/jnp.linalg.norm(y_star)
    x_lst.append(x)
    f_lst.append(jnp.maximum((x[0]**2 + (x[1]+1)**2), (x[0]**2 + (x[1]-1)**2)))
    key, subkey = jax.random.split(key)

# %%
plt.style.use('seaborn-paper')
plt.style.use('bmh')
plt.plot(f_lst, label="danskin")
plt.plot(f_lst_p, label='naive')
plt.legend()

# %%
f_lst

# %%
x = jnp.array([1.0, -1.0 - jnp.sqrt(2)])
key, subkey = jax.random.split(key)
y_star = inner_max(x, 1, 10, subkey)

(y_star), f(x, y_star)

# %%
x_lst = jnp.array(x_lst)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.scatter(x_lst[:, 0], x_lst[:, 1])

# %%
import numpy as np
x_axis = np.linspace(-1.5, 1.5, 50)
y_axis = np.linspace(-1.5, 1.5, 50)

X, Y = np.meshgrid(x_axis, y_axis)

def function(x, y):
    return jnp.maximum((x**2 + (y+1)**2), (x**2 + (y-1)**2))

Z = function(X, Y)



# %%
x_lst = np.array(x_lst)
x_lst_p = np.array(x_lst_p)
#fig, (plt, ax2) = plt.subplots(1, 2,  figsize=(15, 6))
contours = plt.contour(X, Y, Z, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
ddd = plt.scatter(x_lst[:, 0], x_lst[:, 1], label="DDD")
naive = plt.scatter(x_lst_p[:, 0], x_lst_p[:, 1], label="PGD", marker='x')
path=x_lst_p.T
plt.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, width=0.003, alpha=0.5, color=naive.get_facecolor()[0])

path=x_lst.T
plt.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, width=0.003, alpha=0.5, color=ddd.get_facecolor()[0])
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend()
plt.xlabel('Contour plot')

plt.savefig("countour.pdf", dpi=200, format='pdf')

# %%
plt.plot(f_lst, label="DDD")
plt.plot(f_lst_p, label='AT')
plt.xlabel("Iteration")
plt.ylabel("Robust Loss")
plt.legend(fontsize=16)

plt.savefig('functionvalue.pdf', format='pdf', dpi=200)

# %%

# %%
x_lst_p


# %%
def f(x):
    return  ReLU(x) + ReLU(-x)
gf = jax.grad(f)
gf(0.0)


# %%
from jax.nn import relu, elu

def create_params(dimensions, key):
    params = []
    for shape in dimensions:
        key, subkey = jax.random.split(key)
        params.append(0.2*jax.random.normal(subkey, shape))
    return params

def ReLU(x):
    return jnp.maximum(0.0, x)

def sigmoid(x):
    return 1.0/(1.0+jnp.exp(-x))

def network(params, x):
    return sigmoid(jnp.dot(params[2], elu(jnp.dot(params[0], x) + params[1])))

network = jax.jit(jax.vmap(network, in_axes=(None, 0)))

def objective(params, x, y):
    out = network(params, x)
    return jnp.average(-(y*jnp.log(out) + (1-y)*jnp.log(1-out)))


obj_gradient = jax.grad(objective, argnums=(0, 1))

param_grad = jax.jit(lambda p, x, y: obj_gradient(p, x, y)[0])
input_grad = jax.jit(lambda p, x, y: obj_gradient(p, x, y)[1])


# %%
key = jax.random.PRNGKey(0)
network_dimensions = ((2, 2), (2,), (2,))

net_params = create_params(network_dimensions, key)
madry_net = create_params(network_dimensions, key)


def inner_max(params, x, y, delta_0, K, epsilon, step_size):
    delta = delta_0
    for _ in range(K):
        delta = delta + step_size*input_grad(params, x + delta, y)
        delta = jnp.clip(delta, -epsilon, epsilon)
    return delta

w = np.array([1.0] + [0.0])
x = jax.random.normal(key, (100, 2))
y = (np.sign(np.dot(x, w)) + 1.0)/2.0
epsilon = 1


# %%
key = jax.random.PRNGKey(0)
network_dimensions = ((2, 2), (2,), (2,))

net_params = create_params(network_dimensions, key)

danskin_negatives = []
danskin_rob_loss_values = []
danskin_clean_loss_values = []

for k in range(500):
    flat_gradients = []
    gradients = []
    for i in range(10):
        key, subkey = jax.random.split(key)
        delta_0 = epsilon * jnp.sign(jax.random.normal(subkey, x.shape))
        delta_star = inner_max(net_params, x, y, delta_0, 10, epsilon, 0.1)
        gradient = param_grad(net_params, x + delta_star, y)
        flat_gradients.append(jnp.vstack(gradient).reshape(-1))
        gradients.append(gradient)
    
    MMT = (jnp.array(flat_gradients) @ jnp.array(flat_gradients).T)
    danskin_negatives.append(jnp.sum(MMT < 0.0))
    danskin = SimplexMinJax(MMT, 10)
    mixture = danskin.optimize()

    direction = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), net_params)
    for coef, g in zip(mixture, gradients):
        direction = jax.tree_util.tree_map(lambda a, single_grad: a+ coef*single_grad, direction, g)


    danskin_clean_loss_values.append(objective(net_params, x, y))
    delta_star = inner_max(net_params, x, y, delta_0, 100, epsilon, 0.1)
    danskin_rob_loss_values.append(objective(net_params, x + delta_star, y))

    step_size = 0.01
    net_params = jax.tree_util.tree_map(lambda p, grad_p: p - step_size * grad_p/jnp.linalg.norm(grad_p), net_params, direction)
    
    



    

# %%
key = jax.random.PRNGKey(0)
madry_net = create_params(network_dimensions, key)
madry_rob_loss_values = []
madry_clean_loss_values = []
madry_negatives = []

for k in range(250):
    flat_gradients = []
    gradients = []
    for i in range(10):
        key, subkey = jax.random.split(key)
        delta_0 = epsilon * jnp.sign(jax.random.normal(subkey, x.shape))
        delta_star = inner_max(madry_net, x, y, delta_0, 10, epsilon, 0.1)
        gradient = param_grad(madry_net, x + delta_star, y)
        flat_gradients.append(jnp.vstack(gradient).reshape(-1))
        gradients.append(gradient)
    
    MMT = (jnp.array(flat_gradients) @ jnp.array(flat_gradients).T)
    madry_negatives.append(jnp.sum(MMT < 0.0))

    madry_clean_loss_values.append(objective(madry_net, x, y))
    delta_star = inner_max(madry_net, x, y, delta_0, 100, epsilon, 0.1)
    madry_rob_loss_values.append(objective(madry_net, x + delta_star, y))

    step_size = 0.01
    madry_net = jax.tree_util.tree_map(lambda p, grad_p: p - step_size * grad_p/jnp.linalg.norm(grad_p), madry_net, gradients[0])
    
    


# %%
plt.plot(danskin_rob_loss_values, label='DDD')
plt.plot(madry_rob_loss_values, label='PGD')
plt.ylabel("Robust Loss")
plt.xlabel("Iteration")
plt.legend()

plt.savefig('robust_loss.pdf', format='pdf', dpi=200)




# %%
plt.plot(range(len(danskin_negatives)), np.array(danskin_negatives)/2, label="DDD")
plt.plot(range(len(madry_negatives)), np.array(madry_negatives)/2, label="PGD")
plt.ylabel("Negative inner-products count")
plt.xlabel("Iteration")
plt.legend()

plt.savefig("negatives.pdf", dpi=200, format='pdf')

# %%
#key = jax.random.PRNGKey(0)
#madry_net = create_params(network_dimensions, key)

for k in range(1):
    flat_gradients = []
    gradients = []
    deltas = []
    for i in range(100):
        key, subkey = jax.random.split(key)
        delta_0 = epsilon * jnp.sign(jax.random.normal(subkey, x.shape))
        delta_star = inner_max(madry_net, x, y, delta_0, 10, epsilon, 0.1)
        deltas.append(delta_star)
        gradient = param_grad(madry_net, x + delta_star, y)
        flat_gradients.append(jnp.vstack(gradient).reshape(-1))
        gradients.append(gradient)
    
    MMT = (jnp.array(flat_gradients) @ jnp.array(flat_gradients).T)
    print(jnp.sum(MMT < 0.0))
    danskin = SimplexMinJax(MMT, 100)
    mixture = danskin.optimize()

    direction = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), net_params)
    for coef, g in zip(mixture, gradients):
        direction = jax.tree_util.tree_map(lambda a, single_grad: a+ coef*single_grad, direction, g)


gradients.append(direction)
    

# %%
inner_trees(direction,  direction)


# %%
def inner_trees(a, b):
    return jax.tree_util.tree_reduce(lambda u, v: u+v, jax.tree_util.tree_map(lambda x, y: jnp.dot(x.reshape(-1), y.reshape(-1)).item(), a, b), 0.0)


# %%
def substract(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


# %%
losses_along = [[] for _ in gradients]
for i, gradient in enumerate(gradients):
    delta_0 = epsilon * jnp.sign(jax.random.normal(subkey, x.shape))
    current = jax.tree_util.tree_map(lambda u: u.copy(), madry_net)
    #delta_star = inner_max(current, x, y, delta_0, 100, epsilon, 0.1)
    #losses_along[i].append(objective(current, x+delta_star, y))
    losses_along[i].append(max([objective(current, x+d, y) for d in deltas]))
    #losses_along[i].append(max([objective(madry_net, x, y) + inner_trees(g, substract(current, madry_net)) for g in gradients[:-1]]))
    for l in range(100):
        current = jax.tree_util.tree_map(lambda p, grad_p: p - step_size/100 * grad_p/jnp.linalg.norm(grad_p), current, gradient)
        #delta_0 = epsilon * jnp.sign(jax.random.normal(jax.random.PRNGKey(l), x.shape))
        #delta_star = inner_max(madry_net, x, y, delta_0, 1000, epsilon, 0.01)
        #delta_star = inner_max(current, x, y, delta_0, 100, epsilon, 0.1)
        #losses_along[i].append(objective(current, x+delta_star, y))
        losses_along[i].append(max([objective(current, x+d, y) for d in deltas]))
        #losses_along[i].append(max([objective(madry_net, x, y) + inner_trees(g, substract(current, madry_net)) for g in gradients[:-1]]))
    

# %%
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
for i in range(len(gradients)-1):
    plt.plot([k/100*step_size for k in range(101)], losses_along[i], linestyle=linestyle_tuple[i%10][1], label="PGD "+str(i))
plt.plot([k/100*step_size for k in range(101)], losses_along[-1], label='DDD', linewidth=1, color='r')
plt.ylabel("Robust Loss")
plt.legend()
#plt.savefig('mid-linear.pdf', format='pdf')


# %%
plt.bar(range(len(danskin_negatives)), np.array(danskin_negatives)/2, width=4, label="DDD")


# %%
def phi(u):
    return jnp.log(1 + jnp.exp(u))
phi = jax.vmap(phi)

@jax.jit
def linear(x):
    return jnp.average(phi(jnp.dot(A, x)) - b * jnp.dot(A, x))
    #return jnp.sum([phi(jnp.dot(x, a)) - b*jnp.dot(x,a) for a,b in zip(a_lst, b_lst)])

gradlin = jax.jit(jax.grad(linear))

# %%
x = np.random.randn(100)
x_lst = [x]
values = [linear(x)]
for _ in range(100):
    x = x - 0.1*gradlin(x)
    x_lst.append(x)
    values.append(linear(x))

# %%
plt.plot(values)

# %%
x_lin = x[:]


# %%
x_log = x[:]

# %%
plt.scatter(x_lin, x_log, marker='o')
plt.savefig("out.png", dpi=200)

# %%
x_lin

# %%
x_log

# %%
