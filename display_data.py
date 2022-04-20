import matplotlib.pyplot as plt
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def barcharts(title, x, y, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 10))
    species_names = [SPECIES[x1] for x1 in x]
    fig = plt.bar(species_names, y, color='green', width=0.5)
    print_file(filename, title, xlabel, ylabel)

def scatter_species(title, df, x, y, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 10))
    all_species = df.groupby("Species")
    for name, species in all_species:
        plt.plot(species[x], species[y], marker='o', linestyle='', markersize=12, label=SPECIES[name])
    plt.legend(loc='upper left')
    print_file(filename, title, xlabel, ylabel)
    
def print_file(filename, title, xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    filepath = f'../../Images/Unsupervised/{filename}.png'
    plt.savefig(filepath)  
    plt.close()  