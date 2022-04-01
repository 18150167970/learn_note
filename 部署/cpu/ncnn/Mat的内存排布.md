## 1. 读完本章你可以学到的知识点

- 熟悉ncnn::Mat的内存排布

## 2. ncnn::Mat的实现以及内存排布

- 首先, 在c++里我们看到一个类, 一般都想知道这个类是要做什么的, 有什么功能, 简单的说类就是封装了一些函数和一些数据. ncnn::Mat这个类不涉及继承, 它在ncnn里扮演的角色就是**数据的表示.** 通常来说在阅读一个我首次见到的class时, 都是从它的构造函数, 数据成员, 析构函数开始看的.
- 我们先来看构造函数, 构造函数一般来说就是初始化类的数据成员.
- 下面的代码块, 我单独拎出来了其中的一个构造函数, 可以看到这个构造函数的实现是: 先初始化各个数据成员为0, `data(0),refcount(0),elemsize(0),elempack(0),allocator(0),dims(0),w(0),h(0),c(0),cstep(0)`然后调用`create函数`, 再重新初始化这些数据成员.
- 下面分别解释一下各个数据成员的含义
- **data**: 表示Mat分配的内存的头地址, 是一个指针类型
- **refcount**: 表示Mat的引用计数, 是一个指针类型
- **allocator**: 本章我们不太关系这个变量可以认为它的值始终为0, 是一个指针类型
- **dims**: 表示数据的维度, 是一个整数类型
- **w**: 表示数据的width, 是一个整数类型
- **h**: 表示数据的height, 是一个整数类型
- **c**: 表示数据的channel, 是一个整数类型
- **elempack**: 表示有多少个数据打包在一起, 是一个整数类型
- **elemsize**: 表示打包在一起的数据占的字节数, 是一个整数类型
- **cstep**: 表示channel step, 即走一个channel跨过的字节数, 是一个整数类型
- 是不是感觉好复杂, 怎么多的数据成员, 我怎么看的过来呢. 是的初看确实很多, 但是不要慌, 怎么多的变量是为了更加细粒度的控制数据的表示, 到后面你会发现这些数据成员都是那么自然．

```cpp
// the three dimension matrix
class Mat {
public:
    // dim
    Mat(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0) {
    create(_w, _h, _c, _elemsize, _allocator);
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator) {
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;
    release();
    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}
```

- 接下来都以图解的形式, 直观的表示Mat的内存排布

![img](https://pic4.zhimg.com/v2-1becd30819bbf5da70aa2797ff735cd7_b.jpg)

![img](https://pic1.zhimg.com/v2-41db06d3bd9e52ae710dccc31e8a974c_b.jpg)

下面是一个三维的Mat, 先给出实例代码以及对m赋值, 然后图解m的内存具体长什么样

![img](https://pic1.zhimg.com/v2-f7391ae8fb18c3c10d22065e0bf54e84_b.jpg)

![img](https://pic1.zhimg.com/v2-a4c1645385f32aefa722cd90d6961a28_b.jpg)

```cpp
// 返回总共的元素
// 1. 当是3维时, cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
// 这时为了内存对齐, cstep 可能并不等于 w * h, 所以此时total()返回的元素个数大于等于w * h * c

// 2. 当是2维时, cstep = (size_t)w * h, 此时total()返回的元素个数等于w * h * c

// 3. 当是1维时, cstep = w, 此时total()返回的元素个数等于w * h * c
inline size_t Mat::total() const {
    return cstep * c;
}
```

- 第164, 165行, 将一个对象转换为float*, 是c++里`类型转换操作符（type conversion operator）`的语法 

```cpp
// access raw data, 将Mat类型的对象返回一个T*
template<typename T>
operator T*();
```

- 以上分别解释了一维, 二维, 三维的ncnn::Mat的内存排布, 但是它们都是elempack = 1的, 因为有很多小伙伴都不理解elempack这个变量的含义, 下面也给出图解实例, 阅读前请确保理解了上面的讲解

![img](https://pic1.zhimg.com/v2-cd9d72fc0035efb3671a332c2f1f75ec_b.jpg)

------

以上便是ncnn::Mat的内存排布的讲解, 下面是它的析构函数, 值得注意的是`NCNN_XADD(refcount,-1)`是一个宏, 它的作用是使refcount这个指针所指向的值减去一, 不过返回的还是之前的值, 有点类似(i++), 对i加一, 不过返回的是i之前的值.

```cpp
inline Mat::~Mat() {
    release();
}

inline void Mat::release() {
    if (refcount && NCNN_XADD(refcount, -1) == 1) {
        if (allocator) allocator->fastFree(data);
        else fastFree(data);
    }
    data = 0;
    elemsize = 0;
    elempack = 0;
    dims = 0;
    w = 0;
    h = 0;
    c = 0;
    cstep = 0;
    refcount = 0;
}
```

## 3. References

- [Tencent/ncnn](https://link.zhihu.com/?target=https%3A//github.com/Tencent/ncnn/wiki/element-packing) 
- [关于mat中data的内存排列问题 · Issue #334 · Tencent/ncnn](https://link.zhihu.com/?target=https%3A//github.com/Tencent/ncnn/issues/334) 

## 参考文献

https://zhuanlan.zhihu.com/p/336359747