import umap                 # install with: pip install umap

def UnsupervisedUMAP(data):
    model = umap.UMAP(n_components=2)
    umap_data  = model.fit_transform(data)
    #umap_data = umap_data.T # verify dimension of T ??
    return umap_data

if __name__ == "__main__":
    print('main')