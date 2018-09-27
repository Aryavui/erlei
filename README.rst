
erlei
=====

erlei 是一个 python 包，提供函数式编程。

特点
----

-  轻量：< 10KB
-  纯粹：不依赖于其他的第三方包
-  高效：不以牺牲速度来换取便捷
-  Wonderful：A wonderful way for Functional Programming.

安装
----

直接使用 ``pip`` 进行安装：

``pip install erlei``

使用
----

如函数 :math:`f(x, y) = x + y`\ ，在python中，标准写法为：

.. code:: python

    def f(x, y):
        return x + y

使用 erlei 后，这样写：

.. code:: python

    from erlei import _

    f = _ + _

更多功能的使用见下详解。

主要功能
--------

匿名函数（lambda）：新的方式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

正如你在前面的例子中看到的那样，一个下划线“\_”就是一个恒等函数（即 y =
x），下面详细说明“下划线函数”的使用。

在下划线表达式中（也可以将其称为占位符表达式），下划线的数量与自变量的数量一致，参数的顺序与下划线的顺序一致，例如：

.. code:: ipython3

    from erlei import _
    
    d = _**2 + _**2
    d(3, 4)




.. parsed-literal::

    25



第一个下划线对应第一个参数，第二个下划线对应第二个参数，下划线的数量就是参数的数量，上面中dist是一个二元函数，等价于
:math:`f(x, y) = x^2 + y^2`.

我们可以将下划线打印出来看看它究竟为何方神圣：

.. code:: ipython3

    from erlei import _
    
    d = _**2 + _**2
    print(d)


.. parsed-literal::

    <class 'Underscore':a callable object roughly equivalent to function: lambda x1, x2: x1 ** 2 + x2 ** 2>


可见，下划线表达式就是函数，与lambda表达式别无二致，并且无须担心下划线表达式的执行效率。

既然下划线表达式就是函数，那也可以就地调用：

.. code:: ipython3

    (_**2 + _**2)(3, 4)




.. parsed-literal::

    25



更多使用案例：

.. code:: ipython3

    from functools import reduce
    
    reduce(_ + _, range(1, 101))




.. parsed-literal::

    5050



复合函数：管道（pipe）
~~~~~~~~~~~~~~~~~~~~~~

在数学中，函数组合是将一个函数的结果应用于另一个函数以产生第三个函数。例如，函数
:math:`f：X→Y` 和 :math:`g：Y→Z` 可以组合产生一个函数，它将 :math:`X`
中的 :math:`x` 映射到 :math:`Z` 中的 :math:`g(f(x))`\ 。直观地说，如果
:math:`z` 是 :math:`y` 的函数，\ :math:`y` 是 :math:`x` 的函数，那么
:math:`z` 是 :math:`x` 的函数。得到的复合函数表示为
:math:`g∘f：X→Z`\ ，定义为 :math:`(g∘f)(x)= g(f(x))`\ 。

例如：求一个向量的 2-范数
^^^^^^^^^^^^^^^^^^^^^^^^^

    向量的2-范数：向量是一维数组（列表），计算它的每个元素的平方，然后将它们全部加在一起，最后计算它的根。

你可以这样做：

.. code:: ipython3

    import math
    
    vec = [3, 4, 12]
    norm_2 = math.sqrt(sum(map(lambda x: x*x, vec)))
    norm_2




.. parsed-literal::

    13.0



这看起来并不太糟糕。然而，当面对繁琐的情况时，这种地狱式的层次调用看起来不那么友好。使用管道方式，你可以这样做：

.. code:: ipython3

    from erlei import pipe
    
    vec = [3, 4, 12]
    
    norm_2 = pipe >> (lambda v: map(_**2, v))\
             >> sum\
             >> math.sqrt\
             >>print
    
    norm_2 <= vec                               # 将参数扔进管道


.. parsed-literal::

    13.0


上面代码可以看出，将管道 pipe 导入后，直接用代码 ``pipe``
就创建了一个管道，不过该管道不会对数据做任何处理（即是一个恒等变换），接下来可以使用
``>>``
将函数以链的形式链接在管道后，创建成一个更长的、处理能力更强的管道。

创建一个管道以后，可以反复调用该管道，将参数“扔进”管道提供了三种方式，用户根据自己的偏好随便使用某种都行：

1. 函数调用的方式: pipeline(data)
2. 管道操作符: data \| pipeline 或者 pipeline \| data
3. 箭头操作符: ppipelineipe <= data

比如，现在实现一个开方函数，不仅可以对正数开方，负数也能处理（开方前先取绝对值），可以这么做：

.. code:: ipython3

    sqrt = pipe >> abs >> math.sqrt >> print
    
    sqrt <= -16
    -16 | sqrt
    sqrt | -16


.. parsed-literal::

    4.0
    4.0
    4.0


总之，以管道方式创建复合函数是一种绝佳的方式，不仅便捷，代码的可读性也非常高。

.. code:: python

    pipeline = pipe >> func1 >> func2 >> func3

等价于

.. code:: python

    pipeline = lambda x: func3(func2(func1(x)))

实际上，你还可以将第一个函数作为管道的参数，写成：

.. code:: python

    pipeline = pipe(func1) >> func2 >> func3

占位符匿名函数和管道一起使用示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

现有一个包含许多由空格分隔的英语单词的字符串。现在，需要你先获取此字符串中的单词序列，然后过滤掉长度超过3的单词，然后将所有这些单词转换为小写，然后过滤掉以“a”或“s”开头的单词，并按字母顺序排序，最后在屏幕上打印一个单词列表。

看，这过程很是繁琐，但是明显有管道处理的意味，前一个处理过程的结果是后一个处理过程的输入，这就好比一截一截的“管道”连通起来，将污水放入管道，经过每截管道的处理，最后输入干净的水。这里的污水就是包含许多由空格分隔的英语单词的字符串，干净的水就是最终需要的输出结果。

不使用管道技术，你可能会这样做：

.. code:: ipython3

    words = "Arya Sansa Brandon Snow Hodor Lady Ghost Cersei Imp Jaime Renly Joffery"
    print(sorted(filter(lambda s: s.startswith(('a', 's')), map(str.lower, filter(lambda x: len(x) > 3, words.split(" "))))))


.. parsed-literal::

    ['arya', 'sansa', 'snow']


WTF！

是的，它有效，你可以在屏幕上看到结果
``['arya'，'sansa'，'snow']``\ （是的，我喜欢这三个角色）。

您也许认为可以通过这种方式使过程更加优雅（只需调整代码的格式）：

.. code:: ipython3

    words = "Arya Sansa Brandon Snow Hodor Lady Ghost Cersei Imp Jaime Renly Joffery"
    print(
        sorted(
            filter(
                lambda s: s.startswith(('a', 's')),
                map(
                    str.lower,
                    filter(
                        lambda x: len(x) > 3, words.split(" ")
                    )
                )
            )
        )
    )


.. parsed-literal::

    ['arya', 'sansa', 'snow']


你可能已经尽力达到最大程度的易读性，但即使如此，这里还是存在很多嵌套函数，这很糟糕。

我们可以用管道做得更好，根据描述执行此任务分为 6 个步骤：：

1. 获取字符串中的单词列表
2. 过滤掉长度超过3的单词
3. 把所有这些单词变成小写
4. 过滤掉以'a'或's'开头的单词
5. 按字母顺序排序
6. 将它们打印在屏幕上

代码：

.. code:: ipython3

    from erlei import pipe as p
    from functools import partial
    
    
    # Create a pipe to perform this task
    pipe = p >> (lambda li: li.split(" ")) \
         >> partial(filter, p >> len >> (_ > 3)) \
         >> partial(map, str.lower)\
         >> partial(filter, lambda s: s.startswith(('a', 's'))) \
         >> sorted \
         >> print
    
    # Then throw the list into the created pipe
    pipe <= words
    
    # You can use the same pipe to handle another string
    another_words = "Balon Samwell Theon Yara Arynn Jon Lysa Robin Mord Frey Walder Pyp "
    another_words | pipe


.. parsed-literal::

    ['arya', 'sansa', 'snow']
    ['arynn', 'samwell']


讲解一下，为了便捷，导入管道的时候用了 ``as p`` 来使用 ``p``
创建管道。上述过程显然很清晰地看到管道后面链接了 6
个处理过程（函数），每个处理过程独占一行，用 ``\`` 隔开。

此外，链接在管道后面的得是函数（实际上只要是可调用对象都行，多个管道链接在一起也行，其实可以将管道看做函数），所以上面使用了
``partial``
函数，该函数接受一个多参函数和部分参数，返回一个函数，该函数可以接受剩余的函数，例如，\ ``map``
函数接受两个参数，第一个参数是一个函数，第二个参数是被处理的对象，
表达式\ ``partial(map, str.lower)``\ 将函数\ ``str.lower``\ 传给 ``map``
函数，返回另一个函数，该函数接受 ``map``
的第二个参数，然后返回处理结果。

上面有一个管道表达式 ``p >> len >> (_ > 3)``
的结果是一个管道（可调用对象，可以看做函数），该管道接受一个数据，现将该数据作为函数
``len`` 的参数得到长度，后面的占位符表达式 ``_ > 3`` 等价于函数
``lambda x: x > 3``\ 。

.. code:: ipython3

    filter = p >> len >> (_ > 3)
    print("erlei" | filter)


.. parsed-literal::

    True


科里化（currying）
~~~~~~~~~~~~~~~~~~

科里化技术用来处理一个多参函数，跟前面提到的偏函数 ``partial``
十分类似，经过科里化的多参函数，可以传递部分参数，返回一个能接受剩余参数的函数，其与偏函数不同的地方在于，偏函数得接受剩余所有的参数，然后返回函数结果，科里化后的函数不要求一次性接受完剩余所有的参数，举例说明：

.. code:: ipython3

    from erlei.decorators import currying
    
    @currying
    def sum5(a, b, c, d, e):
        return a + b + c + d + e
    
    
    
    print(sum5(1)(2)(3)(4)(5))
    print(sum5(1, 2, 3)(4, 5))
    print(sum5(1, 2)(3)(4, 5))


.. parsed-literal::

    15
    15
    15


.. code:: ipython3

    f = sum5(1, 2)
    g = f(3)
    print(g(4, 5))


.. parsed-literal::

    15


尾递归优化
~~~~~~~~~~

众所周知，递归会嵌套地不断创建栈来保存中间结果，当栈太深的时候，程序可能发生栈溢出二出错。但是，当一些递归是尾递归的形式的时候，可以做优化处理，C++
编译器就做了这个工作，现阶段，Python
还没有针对尾递归做优化处理，那么这里就提供了尾递归优化技术，使得尾递归函数的执行不会无止境地创建新栈而导致栈溢出。

例如，求斐波那契数的函数可以写成尾递归的形式，Python里不做尾递归优化，当栈深超过一定上界，将抛出栈溢出：

.. code:: ipython3

    def fib(i, a=0, b=1):
        if i == 0:
            return a
        else:
            return fib(i - 1, b, a + b)
    
    fib(3000)


::


    ---------------------------------------------------------------------------

    RecursionError                            Traceback (most recent call last)

    <ipython-input-14-8e8742373c91> in <module>()
          5         return fib(i - 1, b, a + b)
          6 
    ----> 7 fib(3000)
    

    <ipython-input-14-8e8742373c91> in fib(i, a, b)
          3         return a
          4     else:
    ----> 5         return fib(i - 1, b, a + b)
          6 
          7 fib(3000)


    ... last 1 frames repeated, from the frame below ...


    <ipython-input-14-8e8742373c91> in fib(i, a, b)
          3         return a
          4     else:
    ----> 5         return fib(i - 1, b, a + b)
          6 
          7 fib(3000)


    RecursionError: maximum recursion depth exceeded in comparison


加上尾递归优化：

.. code:: ipython3

    from erlei.decorators import tail_recurse_optimizer as tro
    
    @tro
    def fib(i, a=0, b=1):
        if i == 0:
            return a
        else:
            return fib(i - 1, b, a + b)
    
    fib(3000)




.. parsed-literal::

    410615886307971260333568378719267105220125108637369252408885430926905584274113403731330491660850044560830036835706942274588569362145476502674373045446852160486606292497360503469773453733196887405847255290082049086907512622059054542195889758031109222670849274793859539133318371244795543147611073276240066737934085191731810993201706776838934766764778739502174470268627820918553842225858306408301661862900358266857238210235802504351951472997919676524004784236376453347268364152648346245840573214241419937917242918602639810097866942392015404620153818671425739835074851396421139982713640679581178458198658692285968043243656709796000



尾递归优化以后，一般创建的栈只是个位数，因此空间复杂度不会增加，计算时间也非常快。

中缀语法
~~~~~~~~

什么是中缀语法呢？比如 ``range(1, 101)`` 是生成 1 到 100
的整数，引入中缀表达后，这样使用：

.. code:: ipython3

    from erlei.decorators import to
    
    print(1 /to/ 100)


.. parsed-literal::

    range(1, 100)


中缀语法将一个二参函数变成一个能用中缀表达式来调用的形式，函数的第一个参数置于左侧，第二个参数置于右侧。

那么，如何将一个二参函数变成一个“中缀函数”呢？

使用 ``infix`` 装饰器即可：

.. code:: ipython3

    from erlei.decorators import infix
    
    @infix
    def plus(a, b):
        return a + b
    
    20 /plus/ 3




.. parsed-literal::

    23



.. code:: ipython3

    is_a = infix(isinstance)
    
    4 /is_a/ int




.. parsed-literal::

    True



.. code:: ipython3

    @infix
    def drop(obj, n):
        return obj[n:]
    
    [1, 2, 3, 5, 6, 7, 8] /drop/ 3




.. parsed-literal::

    [5, 6, 7, 8]



.. code:: ipython3

    @infix
    def take(obj, n):
        return obj[:n]
    
    [1, 2, 3, 5, 6, 7, 8] /take/ 3




.. parsed-literal::

    [1, 2, 3]



实际上 ``erlei.decorators`` 提供了这些预置的中缀语法：\ ``is_a``,
``to``, ``take``, ``drop``, ``has`` 等。

例如：

.. code:: ipython3

    from erlei.decorators import to, step, has
    
    print(1 /to/ 11 /step/ 2)
    
    class Point:
        x = 0
        y = 0
        
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    
    p = Point(3, 4)
    print(p /has/ 'x')


.. parsed-literal::

    range(1, 11, 2)
    True


Enjoy it!
=========
