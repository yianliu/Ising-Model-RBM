data_path = 'Training Data'
Engs = []
Mags = []
for i in range(nt):
    T = T_range[i]
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeName = os.path.join(data_path, file_name)
    samples = np.load(completeName)
    Engs.append(E(samples))
    Mags.append(M(samples))
E_vals = []
E_errs = []
for i in Engs:
    E_vals.append(i[0])
    E_errs.append(i[1])


plt.errorbar(T_range, E_vals, yerr = E_errs)

plt.title('matplotlib.pyplot.errorbar() function Example')
plt.show()
M_vals = []
M_errs = []
for i in Mags:
    M_vals.append(i[0])
    M_errs.append(i[1])


plt.errorbar(T_range, M_vals, yerr = M_errs)

plt.title('matplotlib.pyplot.errorbar() function Example')
plt.show()
