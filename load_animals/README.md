This is an alternative to the load_data in the homework.
It can be used with:

instead of
```
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
```
we can use
```
train_x_orig, train_y, test_x_orig, test_y, classes = load_random_animals(
    dataset_folder = 'root_folder_of_the_dataset',
    max_classes=2,
    img_size=(64,64),
    train_test_split=0.8)
```
and it should load `max_classes` random classes.
In this case, `img_size` is the size the image is resized to (very small, but since we convert this to vectors, it is easier for now and for performance reasons) and `train_test_split` is the ratio in the dataset splitting.
