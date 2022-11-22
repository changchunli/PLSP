from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        assert X.shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])


class labeled_dataset(Dataset):
    def __init__(self, dataset, pseudo_labels, indexs, transform=None):
        self.images = dataset.images[indexs]
        self.given_label_matrix = dataset.given_label_matrix[indexs]
        self.true_labels = dataset.true_labels[indexs]
        self.pseudo_labels = pseudo_labels[indexs]
        self.transform = transform

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        each_pseudo_label = self.pseudo_labels[index]

        if self.transform is not None:
            each_image = self.transform(each_image)

        return each_image, each_label, each_pseudo_label, each_true_label, index


class unlabeled_dataset(Dataset):
    def __init__(self, dataset, indexs, transform=None):
        self.images = dataset.images[indexs]
        self.given_label_matrix = dataset.given_label_matrix[indexs]
        self.true_labels = dataset.true_labels[indexs]
        self.transform = transform

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        if self.transform is not None:
            each_image = self.transform(each_image)

        return each_image, each_label, each_true_label, index


class gen_index_dataset(Dataset):
    def __init__(self,
                 images,
                 given_label_matrix,
                 true_labels,
                 transform=None):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.transform = transform

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        if self.transform is not None:
            each_image = self.transform(each_image)

        return each_image, each_label, each_true_label, index
