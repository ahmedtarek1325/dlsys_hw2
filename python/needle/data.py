from random import shuffle
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import gzip
import struct
from collections.abc import Iterable

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img: 
            return img[:,::-1,:]
        else: 
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        # boundary condition if no padding then 
        # there will not be shift. Consequntly no crops
        if self.padding ==0 : 
            return img


        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        
        # doing the padding via intializaing an empty Tensor
        # and then setting the center of the tensor with the image 
        shape_ = np.array(img.shape) + np.array((self.padding*2,self.padding*2,0))
        padded_img= np.zeros(tuple(shape_))
        padded_img[self.padding:-self.padding,self.padding:-self.padding,:] = img


        ########################################################
        #           FOR THE CROPPING                            #
        #               PART                                    #
        #########################################################

        # if padding equals either shiftx or shift y 
        # then then end of the slicing will = zero and hence
        # we'll recieve an error. To avoid this, I've put boundary
        # conditions checkers
        if shift_x == self.padding and shift_y==self.padding: 
            return padded_img[self.padding+shift_x:,\
            shift_y+self.padding:,:]
        elif shift_x == self.padding: 
            return padded_img[self.padding+shift_x:,\
            shift_y+self.padding:shift_y-self.padding,:]
        elif shift_y == self.padding : 
            return padded_img[self.padding+shift_x:shift_x-self.padding,\
            shift_y+self.padding:,:]  

        # if the self.padding not equal to shift_X or shift_y
        # then we want to crop from the boundariers of the image with 
        # the shifts amounts
        return padded_img[self.padding+shift_x:shift_x-self.padding,\
            shift_y+self.padding:shift_y-self.padding,:]    

        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.current_batch = 0
        indices = list(range(len(self.dataset)))
        if self.shuffle: 
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.current_batch >= len(self.ordering):
            raise StopIteration
        
        '''batch_img,batch_label = [],[]
        for i in self.ordering[self.current_batch]:
            img,label = self.dataset[i] 
            batch_img.append(img)
            batch_label.append(label)'''

        batch = [self.dataset[i] for i in self.ordering[self.current_batch]]
        self.current_batch += 1 
        
        return tuple(Tensor.make_const(np.stack(b)) for b in zip(*batch))
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    import gzip
    import struct
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.y=  self.reading_label_name(label_filename)
        self.X= self.reading_imgaes(image_filename)
        self.transforms = transforms
        self.len = len(self.y)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION 
        y= self.y[index]
        single= True
        if isinstance(y, Iterable): 
            if len(y)> 1 : 
                single =False

        if single: 
            X= self.X[index,:].reshape(28,28,1)
            X = self.apply_transforms(X)
            X=X.reshape((-1))
            return (X,self.y[index])
        else: 
            Xs= self.X[index].reshape(-1,28,28,1)
            Xs = np.array([self.apply_transforms(X).reshape((-1,)) for X in Xs])
            return (Xs,y)


        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.len
        ### END YOUR SOLUTION
    def reading_label_name(self,label_filename): 
        with gzip.open(label_filename, 'rb') as f:
            magic,size= struct.unpack(">II",f.read(8))
            y = np.array(list(f.read()),dtype=np.uint8)
            y = y.reshape((size,)) 
            return y
    def reading_imgaes(self,image_filesname): 
        with gzip.open(image_filesname,'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            X = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
            X = X.reshape((size, nrows*ncols))
            X=X.astype(np.float32)
            X/=255
            return X

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
