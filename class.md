# In class

## from Cewu

text region image - to - text color distance

把字与他周围的部分抠出来，region的距离，d_0, d_1, ..., d_n

we want to maximize min[d_0,... ,d_10]

- input an image, input an subtitle color, input an order, get a distance(we can change the order)
- we have c_1, c_2 ... for every frame, randomly pick 3000 color,
  then we have different d for every random picked color c',
  do a dynamic programming on cost of transition

## our discussion

random pick color, choose from a color range that user can specify
