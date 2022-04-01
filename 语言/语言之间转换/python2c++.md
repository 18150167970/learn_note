

## Python向C++转换和优化的经验分享

> 作者：林建强（linjianqiang@star-net.cn）

- Python向C++转换和优化的经验分享
  - [1 背景](https://wiki.starnetiot.net/read/Python_To_CPP/blank#1 背景)
  - 2 解决方案
    - 2.1 自动转换器
      - [2.1.1 Cython](https://wiki.starnetiot.net/read/Python_To_CPP/blank#2.1.1 Cython)
      - [2.1.2 Pythran](https://wiki.starnetiot.net/read/Python_To_CPP/blank#2.1.2 Pythran)
    - 2.2 手动翻译
      - [2.2.1 数据类型对应](https://wiki.starnetiot.net/read/Python_To_CPP/blank#2.2.1 数据类型对应)
      - [2.2.2 矩阵结构设计](https://wiki.starnetiot.net/read/Python_To_CPP/blank#2.2.2 矩阵结构设计)
      - [2.2.3 转化示例](https://wiki.starnetiot.net/read/Python_To_CPP/blank#2.2.3 转化示例)
  - 3 程序优化
    - [3.1 NEON加速](https://wiki.starnetiot.net/read/Python_To_CPP/blank#3.1 NEON加速)
    - 3.2 几种通用的优化方法
      - [3.2.1 避免重复计算](https://wiki.starnetiot.net/read/Python_To_CPP/blank#3.2.1 避免重复计算)
      - [3.2.2 查表代替计算](https://wiki.starnetiot.net/read/Python_To_CPP/blank#3.2.2 查表代替计算)
      - [3.2.3 变换计算方式](https://wiki.starnetiot.net/read/Python_To_CPP/blank#3.2.3 变换计算方式)
      - [3.2.4 编译选项优化](https://wiki.starnetiot.net/read/Python_To_CPP/blank#3.2.4 编译选项优化)
  - 4 总结

### 1 背景

------

得益于良好的语言设计、完备的第三方库支持，Python在人工智能领域攻城拔寨，大杀四方。感知仓项目中应用到的各类AI算法均由Python作为开发语言进行设计，然而受制于本就不富裕的存储空间以及Python略显缓慢的执行速度，我们只能含泪放弃在设备上直接部署Python AI程序。怎么办？自然地我们会想到把Python程序转成C/C++程序。

### 2 解决方案

------

#### 2.1 自动转换器

人类的惰性让我们首先考虑是否有现成的工具可以实现Python到C/C++的转化？还真有，比如：Cython与Pythran。

##### 2.1.1 Cython

Cython是一种相对成熟的将Python转换为C的编译器，它在Python语法基础上添加一些静态语言的特性（比如确定值类型等），通过一个Python/C API来编译成C代码。这种方法生成的C代码执行速度相较于Python虽有提高，但仍低于原生C，且生成的代码晦涩难懂可读性差，不易于维护。另外Python的有些语法特性它不支持，会有绕不过去的坑，感兴趣的同学可以猛戳这里：[Cython的用法以及填坑姿势](https://blog.csdn.net/feijiges/article/details/77932382)。

##### 2.1.2 Pythran

Pythran是新兴的比较热门的开源项目，能够实现一条命令全自动把Python翻译成等价C++。Pythran支持的Python语言特性子集有：

- polymorphic functions 多态函数(翻译成 C++ 的泛型模板函数)
- dictionary, set, list 等数据结构，map, reduce 等函数
- exceptions异常
- file handling 文件处理
- 部分 Numpy

不支持的功能包括：

- classes类
- polymorphic variables 可变类型变量

细心的同学可能注意到了，Pythran只支持部分Numpy，比如对numpy.reshape、numpy.ndarray的操作等支持不够好。然而Numpy对机器学习的重要性不言而喻，像我们使用的YOLO、Retinaface等算法均会频繁使用reshape等Numpy操作。另外Pythran与Cython一样，生成的C++运行效率依然比不上原生C++。

现成工具均不甚理想，计将安出？

#### 2.2 手动翻译

放弃工具幻想，我们选择看起来最原始却也行之有效的办法：手动逐句翻译，并设计一套结构代替Numpy的部分功能。

由于Python和C++都是面向对象型语言，其基本语法逻辑是共通的，结合感知仓AI算法的设计，我们主要考虑的是两者数据类型的转换以及如何代替Numpy中矩阵类型的计算。

##### 2.2.1 数据类型对应

Python有五个标准数据类型

- Numbers（数字）
- String（字符串）
- List（列表）
- Tuple（元组）
- Dictionary（字典）

Python3中多了一个Set（集合）类型。在这些类型中，部分类型可以在C++中找到对应的，例如Numbers、String可以用int、float、double、bool、char、string来代替。部分类型诸如List、Tuple及Dictionary无法在C++中找到对应的类型，但是没关系，我们可以通过灵活运用数组array、结构体struct、共用体union、容器vector及map等来实现相应的类型。举个Dictionary的例子：

```
anchorDict = {'base':16, 'scale':8, 'stride':4}
```

这里我们可以用C++的unordered_map容器来翻译：

```
std::unordered_map<string, int> anchorDict{{"base", 16}, {"scale", 8}, {"stride", 4}};
```

也可以用一个包含pair类型元素的vector向量来转化：

```
std::vector<std::pair<string, int>>anchorDict{{"base", 16}, {"scale", 8}, {"stride", 4}};
```

还可以直接用静态数组来表达：

```
typedef struct{    string s;    int val;}DICT_ELE;DICT_ELE anchorDict[3];anchorDict[0].s = "base";anchorDict[0].val = 16;anchorDict[1].s = "scale";anchorDict[1].val = 8;anchorDict[2].s = "stride";anchorDict[2].val = 4;
```

实际运用哪种方法，就需要根据算法的具体情况进行选择了。以上的转换例子是为了说明在翻译过程中，不必拘泥形式，死板直译，我们完全可以根据C++自身特性，经过适当变形、组合来达到程序移植的目的。

##### 2.2.2 矩阵结构设计

AI算法中基本绕不开使用Numpy进行数值计算，实际应用中我们完全可以把它看做是一种支持各种维度的数组及矩阵计算工具。通过代码分析发现，一个四维的矩阵足以代替项目中用到的Numpy相关计算。

这个时候追求效率的小伙伴们又想到C++是否有满足条件的矩阵库支持？糟心的是，诸如Matlab C++库、Eigen3、Boost等工具，要么是二维矩阵运算库，要么未实现transpose等一些AI算法基本操作。

因此决定还是乖乖设计一个四维矩阵来代替吧。这个过程中我们参照了opencv的cv::Mat类，设计了一个如下所示的四维矩阵Matrix4：

```
class Matrix4 {private:    int x_rows,y_rows,z_rows,t_rows;    int prealloc;    float *num;public:    Matrix4();    Matrix4(int, int, int, int);    Matrix4(int, int, int, int, float *);    ...    Matrix4(const Matrix4&);    Matrix4(Matrix4&&);    ~Matrix4();    void reshape(...);    void transpose(...);    void argmax(...);    void sigmoid(...);    void slice(...);    /*我们是一系列对应于Numpy矩阵计算的接口^_^*/    ...}；
```

其中，x_rows, y_rows, z_rows, t_rows分别表示矩阵的4个维度。

num则是用来存放矩阵元素的空间地址，它可以是矩阵自身动态申请，也可以是外部预先分配的空间。

我们还提供了一个移动构造函数Matrix4(Matrix4&&)，搭配std::move()使用，在某些情况下可以达到数据“零拷贝”的效果。

接下来所有矩阵的数值计算都可以通过操作这仅有的少数几个属性来完成，一些接口也会变得简单且快速，例如：

矩阵元素的get与set

```
void Matrix4::set_element_value_idx(int idx, float value)    {    num[idx] = value;}
```

只要给出指定元素的索引，比如A[2,3,4,5]，就可以很容易计算并找到存放该元素的偏移地址，随后进行取值或赋值。 $$ idx=2*3*4*5+3*4*5+4*5+5 $$ 矩阵形状reshape

```
void Matrix4::reshape(int x, int y, int z, int t){    if(x_rows*y_rows*z_rows*t_rows != x*y*z*t)    {        printf("dim illegal!\n");        return;    }    x_rows = x;    y_rows = y;    z_rows = z;    t_rows = t;}
```

在这种结构下进行矩阵的reshape操作，完全不用改变矩阵数据空间的元素值或元素存放位置，直接修改四个维度即可，十分方便。

##### 2.2.3 转化示例

我们截取了感知仓中人脸检测Retinaface模型输出后处理的部分Python源码如下：

```
for _idx,s in enumerate(self._feat_stride_fpn):    _key = 'stride%s'%s    stride = int(s)    if self.use_landmarks:      idx = _idx*3    else:      idx = _idx*2    scores = net_out[idx].asnumpy()    scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]    idx+=1    bbox_deltas = net_out[idx].asnumpy()    height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]    A = self._num_anchors['stride%s'%s]    K = height * width    anchors_fpn = self._anchors_fpn['stride%s'%s]    anchors = anchors_plane(height, width, stride, anchors_fpn)    anchors = anchors.reshape((K * A, 4))    scores = self._clip_pad(scores, (height, width))    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))    bbox_deltas = self._clip_pad(bbox_deltas, (height, width))    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))    bbox_pred_len = bbox_deltas.shape[3]//A    bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))    proposals = self.bbox_pred(anchors, bbox_deltas)    proposals = clip_boxes(proposals, im_info[:2])    scores_ravel = scores.ravel()    order = np.where(scores_ravel>=threshold)[0]    proposals = proposals[order, :]    scores = scores[order]    if stride==4 and self.decay4<1.0:      scores *= self.decay4    proposals[:,0:4] /= im_scale    proposals_list.append(proposals)    scores_list.append(scores)
```

这段Python程序主要是对模型输出的9组矩阵数据进行加工，得到人脸框的坐标、置信度及特征点信息。利用前文所述的方法进行转换，可以得到如下所示C++代码：

```
for(int _idx = 0; _idx < stride_fpn_size; _idx++){    int idx = 0;    int stride = m_feat_stride_fpn[_idx];    int _key = stride;    if(m_use_landmarks)    {        idx = _idx * 3;    }    else    {        idx = _idx * 2;    }    Matrix4 scores_init;    scores_init = net_out[idx];    int start, end;    start = 2;     for(auto x_num : m_num_anchors)    {        if(_key == x_num.stride)        {            start = x_num.num;            break;        }    }    end = scores_init.shape(1);    int height = scores_init.shape(2);    int width = scores_init.shape(3);    scores_init.slice_dim2(start, end);    _clip_pad(scores_init, height, width);    scores_init.transpose_0231();    int scores_mat_len = scores_init.shape(0) * scores_init.shape(1) * scores_init.shape(2) *                             scores_init.shape(3);    scores_init.reshape(1,1,scores_mat_len,1);    std::vector<int> order;    float score_tmp;    for(int i = 0; i < scores_mat_len; i++)    {        score_tmp = scores_init.num[i];        if(score_tmp >= threshold)        {            order.push_back(i);            if(4 == stride && m_decay4 < 1.0)            {                scores_list.push_back(m_decay4*score_tmp);            }            else            {                scores_list.push_back(score_tmp);            }        }    }    if(0 == order.size())    {        continue;    }    idx += 1;    Matrix4 bbox_deltas_init;    bbox_deltas_init = net_out[idx];    int A = start;    int base_anchors_idx = 0;    for(auto an_fpn : m_anchors_fpn)    {        if(_key == an_fpn.stride)        {            break;        }        base_anchors_idx++;    }    Matrix4 anchors(height, width, m_anchors_fpn[base_anchors_idx].anchors.size(), 4);    anchors_plane(height, width, stride, m_anchors_fpn[base_anchors_idx].anchors, anchors);    int anchors_mat_len = anchors.shape(0)*anchors.shape(1)*anchors.shape(2)*anchors.shape(3);    anchors.reshape(1, 1, anchors_mat_len/4, 4);    _clip_pad(bbox_deltas_init, height, width);    bbox_deltas_init.transpose_0231();    int bbox_pred_len = bbox_deltas_init.shape(3) / A;    int bbox_deltas_mat_len = bbox_deltas_init.shape(0) * bbox_deltas_init.shape(1) *                                 bbox_deltas_init.shape(2) * bbox_deltas_init.shape(3);    bbox_deltas_init.reshape(1,1,bbox_deltas_mat_len/bbox_pred_len, bbox_pred_len);    bbox_pred_clip_order(anchors, bbox_deltas_init, proposals_list, order);    if ((!m_vote) && m_use_landmarks)    {        idx += 1;        Matrix4 landmark_deltas_init;        landmark_deltas_init = net_out[idx];        _clip_pad(landmark_deltas_init, height, width);        int landmark_pred_len = landmark_deltas_init.shape(1) / A;        landmark_deltas_init.transpose_0231();        int lm_mat_len = landmark_deltas_init.shape(0) * landmark_deltas_init.shape(1) *                             landmark_deltas_init.shape(2) * landmark_deltas_init.shape(3);        landmark_deltas_init.reshape(1, lm_mat_len/landmark_pred_len, 5, landmark_pred_len/5);        landmark_pred_order(anchors, landmark_deltas_init, landmarks_list, order);    }}
```

对比这两段程序可以发现，除了对原始Python代码作了合理的C++等效之外，我们还调整了部分数据的处理逻辑，优化了人脸检出的速度。这也印证了前文所强调的，转化过程需因势利导，灵活应对。

### 3 程序优化

------

由于缺少了Numpy的支持，我们必须通过不断地优化Matrix4来弥补无法使用这个超级强大的Python库造成的损失。

#### 3.1 NEON加速

Matrix4中的sigmoid接口，是深度神经网络中经常会用到的操作，其计算公式如下： $$ S(x) = 1 / (1 + exp(-x)) $$ 可以看到该公式中使用了幂指数函数exp，一般情况下我们使用C的库函数exp()来进行计算，然而它精度虽高，速度却很慢，且实际应用中幂指数通常都是浮点型，真是雪上加霜啊。不过没关系，这里给出一种单精度浮点型exp快速算法：

```
float fastexp(float x){    union {uint32_t i;float f;} v;    v.i=(1<<23)*(1.4426950409*x+126.93490512f);    return v.f;}
```

只需要进行两次浮点乘法运算，便能以极小的精度损失为代价，换取极大的运算速度提升。该算法的基本原理是利用了浮点型变量在内存中的布局从而进行巧妙的运算操作，具体设计原理传送门：[快速浮点数exp算法](https://blog.csdn.net/shakingWaves/article/details/78450500?locationNum=6&fps=1)。

寻得利器，暗自欢喜，止步于此？不不不，考虑到感知仓是跑在ARM平台上的，我们注意到ARM的一个神器NEON。NEON是适用于ARM Cortex-A系列处理器的一种128位SIMD扩展结构，用好它的intrinsic指令集（[ARM Neon Intrinsics各函数介绍](https://blog.csdn.net/hemmingway/article/details/44828303)）可以大大提高我们的计算速度。

NEON的优势是并行计算，所以我们对一个数组的每一个元素进行exp并相加，然后将其加速：

```c++
void Matrix4::matexp(){    
    int i;    
    int len = x_rows*y_rows*t_rows*z_rows;    
    float32x4_t sum_vec=vdupq_n_f32(0);    
    float32x4_t ai=vdupq_n_f32(1064807160.56887296), bi;    
    nt32x4_t int_vec;    
    for(i=0;i<len-4;i+=4)    
    {        
        bi = vld1q_f32(&num[i]);        
        sum_vec=vmlaq_n_f32(ai,bi,12102203.1616540672);        
        int_vec=vcvtq_s32_f32(sum_vec);        
        sum_vec=vreinterpretq_f32_s32(int_vec);        
        vst1q_f32(&num[i],sum_vec);    
    }    
    if(i<len)    
    {        
        float *p = &num[i];        
        for(;i<len;i++)        
        {            
            *p = fastexp(*p);            
            p++;        
        }    
    }
}
```

在前面提到的快速算法中是先计算(1<<23)，然后将其和另外一部分相乘，我们将其简化成一个乘加操作： 

 **12102203.1616540672*x+1064807160.56887296**算法先加载4个变量，然后执行乘加操作。之后的操作首先是将float类型的变量转成int型变量，之后再通过地址强转获取float值并累加。相比原始的exp累加，速度能有5、6倍左右的提升。

除此之外，我们还可以将矩阵的行列求和、矩阵部分转置操作等使用指令集进行优化，使程序获得更多加速。

#### 3.2 几种通用的优化方法

##### 3.2.1 避免重复计算

```
x = a*b+c;
y = a*b-d;
```

可以优化为

```
tmp = a*b;
x = tmp+c;
y = tmp-d;
```

别小看这一改动，在例如高分辨率的图像矩阵数据循环处理中，这个方法可能可以帮你节省数以万计的浮点数乘除法运算！

##### 3.2.2 查表代替计算

在处理器资源紧张而存储器资源相对富裕的情况下，可以用牺牲存储空间换取运行速度的办法。例如需要频繁计算正弦或余弦函数值时，可预先将函数值计算出来置于内存中以供查找。

##### 3.2.3 变换计算方式

(1)移位代替乘除

通常情况下乘以或除以2 的N次方都可以通过左移或右移N位来完成。实际上乘以任何一个整数都可以用移位和加法来代替乘法。例如：

```
i = i * 5;
```

我们可以写为

```
i = (i<<2) + i;
```

(2)与代替求余

有时可以通过用与指令（&）代替求余操作（% ）来提高效率。例如：

```
i = i % 8;
```

可替换为

```
i = i & 0x07;
```

(3)乘法代替除法/乘方

ARM内建有乘法器，因此可以通过乘法来代替除法或乘方以节约函数调用的开销。例如：

```
if(x/y > z)    {x = pow(x, 2.0);}
```

可变通为

```
if(x > y*z)    {x = x*x;}
```

##### 3.2.4 编译选项优化

多数编译器都支持对程序速度和程序大小的优化，有些编译器还允许用户选择可供优化的内容及优化的程度。相比前面的各种优化方法， 有时候通过设置编译器选项对程序进行优化不失为一种简单有效的途径。

### 4 总结

------

感知仓AI算法移植过程中，需要将Python程序转换为C++，目前的自动翻译工具有局限，日后随着它们的完善可能重新为你我所用。在我们手动移植的过程中，需要做到闪展腾挪，以灵活手段扫平崎岖道路。最后我们要注意的是，程序的优化通常只是我们软件设计需要达到的诸多目标之一，优化应在不影响程序正确性、健壮性、可移植性及可维护性的前提下进行。正所谓物极必反，片面追求程序的优化往往会影响其健壮性、可移植性等重要目标，小伙伴们要注意拿捏好尺度呀。

# 参考文献 #

https://wiki.starnetiot.net/read/Python_To_CPP/blank