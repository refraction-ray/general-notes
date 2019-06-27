[TOC]

## 1. QKD 方案

   1. BB84  方案：

      Alice 制备一组量子态，制备时随机选取为 X 测量基或 Z 测量基下的本征态  $\vert \pm x\rangle$,  $\vert \pm z\rangle$ ，并将这些态发送给 Bob。Bob 则随机选择 X 测量基或 Z 测量基进行测量。之后，双方通过经典信道公开各自的测量基序列，二者测量基一致的部分的结果即可作为共享的密钥。

      窃听检测：在测量基相同的部分中，选出一小部分结果公开并比对。如果存在 Eve，由于其无法提前知道 Alice 的制备基，比对结果中会有 1/4 的结果 Alice 与 Bob 不一致，由此可发现窃听。

   2. B92 方案：

      Alice 制备一组量子态，随机制备为 $\vert +x\rangle$ 作为 0 或者 $\vert +z\rangle$ 作为 1，并发送给 Bob。Bob 依旧随机选取 X 或者 Z 测量基测量。注意到，此时如果在 Z 或 X 测量基下测的本征值为 -1，则 Bob 知道对应的态一定为 0 或者 1，否则记为 $?$。结束后只需要 Bob 告诉 Alice 自己得到的确定态的序号，二者即可将这些态作为共享密钥。

      窃听检测：A，B 双方公开部分 Bob 指定的得到确定结果的非 $$?$$ 测量，如果存在 Eve，则两者会存在一定比例的结果不一致。该错误率为 1/4。这一概率，需要详细列出包含 A 发送态，E 测量基，E 测量结果（E 发送给 B 的态可能需要根据测量结果修改），B 测量基，B 测量结果的五层二叉树，以最后结果中 B 确认的态的数目为分母，结论与 A 态不符的部分为分子得出。其次，在此之前还需注意，Bob 可确定的态（非$?$）的比例应占 1/4 左右，否则可能已存在窃听者篡改。

      *PhysRevLett.68.3121*

   3. E91 方案：

      将一系列 EPR pair （$\vert 01\rangle+\vert 10\rangle$） 从一个源分别发送给 Alice 和 Bob。A,B 都随机选择测量基测量，我们将其测量方向记为 $\vec{a_i}, \vec{b_i}, i=1,2,3$。其中 Alice 的测量方向分别是 0，45，90 度，而 Bob 的测量方向分别是 45，90，135 度（以垂直粒子运动的平面为基准）。测量完成后，双方公开测量基，测量基相同（a2和b1, a3和b2）的部分，测量结果必定相反，由此可作为密钥。

      窃听检测：我们定义 $E(\vec{a_i},\vec{b_j})=P_{++}+P_{--}-P_{+-}-P_{-+}$，该量描述了两个测量方向的相关性。由量子力学可知 $E(\vec{a_i},\vec{b_j})=-\vec{a_i}\cdot\vec{b_j}$。定义 $S=E(a_1,b_1)-E(a_1,b_3)+E(a_3,b_1)+E(a_3,b_3)$，对于完全纠缠态，我们有 $E=-2\sqrt{2}$。 测量基选择不同的部分，可用来计算 S 是否违背贝尔不等式，从而确保 EPR pair 的纠缠依旧充分，保证了无窃听者替换了粒子。也即如果有窃听者测量，干扰纠缠态的话，则 $\vert S\vert\leq \sqrt{2}$，满足Bell 不等式定域性。

      *PhysRevLett.67.661*

## 2. Dense Coding

**实现**：Alice 首先将 EPR 对中的一个粒子发送给 Bob。此时 Alice 对自己的粒子做 I，X，Y，Z 操作中的一个。则相应的原始的 $\vert 00\rangle+\vert 11\rangle$ 的态对应转化为 $\vert 00\rangle+\vert 11\rangle$, $\vert 10\rangle+\vert 01\rangle$, $\vert 10\rangle-\vert 01\rangle$，$\vert 00\rangle-\vert 11\rangle$。操作后， Alice 将该粒子也发送给 Bob。注意到此过程，只传递了一个 qubit（第一次的准备工作不算）。而 Bob 最后收到全部两个粒子后，可以进行 Bell 基测量，从而知道对应的态，也即 Alice 的 I，X，Y，Z 信息，恰好为两个经典比特。由此实现了用一个 qubit 进行两个经典 bit 的密集编码。

**用途**：dense coding 可以作为量子直接通讯 (QSTC) 的 subroutine。

**补充**：注意到还有和密集编码过程相反的 quantum teleportation。这一过程通过两个经典 bit 的信息，来传递一个 qubit 信息，将一个量子波函数传递给 Bob。

   * 量子密集编码线路：

   ![densecoding](https://user-images.githubusercontent.com/35157286/59329951-b36a3f80-8d22-11e9-9855-7727a39401cf.png)

   * 量子隐形传态线路：

   ![Quantum-Circuit-for-Quantum-Teleportation](https://user-images.githubusercontent.com/35157286/59329802-5a9aa700-8d22-11e9-9aac-b2b1a44ddab3.png)



## 3. 量子计算普适门集合

精确：CNOT 加所有单比特门

任意精度近似：CNOT，H，S，T

## 4. 量子安全直接通讯方案

QSDC 无需首先通过量子信道协商和确认密钥再在经典信道加密传输信息，而是直接将信息本身在量子信道内传输。

方案：Alice 制备一组 EPR pairs，并把每对中的一个粒子发送给 Bob。Bob 接受完毕后，Alice 选择其中的部分 pair，双方将这部分粒子在同一组基下测量，并比对结果，作为第一次窃听检测。之后 Alice 利用密集编码的方式，对于在自己这边的粒子，选择作用 I，X，Y，Z 操作中的一个，直接将编码的信息包含其中。需要注意 Alice 需要随机挑选一些粒子，编入无用信息，以进行第二轮窃听检测。并之后把所有粒子发给 Bob。Bob 进行 Bell 基测量得出信息。此时进行第二次窃听检测，由 Alice 通知 Bob 检测位，双方比对这些无用信息的结果是否吻合。

安全性分析：如果第一批 Alice 发送给 Bob 的粒子被 Eve 截获，那么在第一次测听检测时，Alice 和 Bob 的测量结果无法总是相等。如果第二批发送给 Bob 的粒子被 Eve 截获，那么Bob Bell 基测量的结果也无法总是和 Alice 编码的信息相等。更进一步地，如果 Eve 无法同时获取两批发送的粒子，就无法通过 Bell 基测量来获取信息。

*PhysRevA.65.032302*

## 5. 量子秘密共享方案

QSS 可使得只有几方同时认同参与才能使用完整的密钥。比如通过协商的方式，Alice 可以保证自己的密钥，只有 Bob 和 Charlie 同时同意时才能使用。也即该密钥可由 Alice 使用，也可以由 Bob+Charlie 使用。

首先 Alice 制备一组 GHZ 态，也即 $$\vert GHZ\rangle=\frac{1}{\sqrt{2}}(\vert 000\rangle +\vert 111\rangle)$$。Alice 将每组三个粒子中的两个分别发送给 B 和 C。之后对于每组态，ABC 随机选取 X 或 Y 基测量，并之后公布测量基。对于 8 种测量基选取情况，有 4种情形，BC 可根据两者测量结果结合，推测出 A 的测量结果。测量结果有关联性的四组基选取为 XXX, XYY, YYX, YXY。其中对应了三者测量结果需要相同或者相反的关联性。

窃听检测：所有测量结束后，ABC 随机选取一部分应该有关联性的测量结果进行比对，如果关联性与理论不符，则存在窃听。

*PhysRevA.59.1829*

## 6. Shor 算法

请先参考问题 12 相位估计算法，其为 Shor 算法的重要 subroutine。

Shor 算法是量子部分解决阶问题和经典部分转化为合数分解问题的结合，其总体上可以在多项式时间解决合数分解问题，实现可能的指数加速。

阶数问题：设正整数 $$x<N$$，x，N 互质，定义阶数 r 为最小的正 r 使得 $$x^r\equiv 1 \mod\; N$$. Shor 算法的说明因此分为两部分，首先我们给出解决阶数问题的多项式量子线路；之后，我们说明阶数问题与合数分解等价，很容易通过经典算法转化。

定义 Unitary operator $$U_x\vert y\rangle=\vert xy\; \mod N\rangle$$。 阶数问题实际就是对 $$U_x$$ 做相位估计。这里的 $$y<N$$，若 $$y\geq N$$，可定义对应部分的 U 算符为 identity。该矩阵是 Unitary 是由 $$xy\;\mod N$$ 对于 $$y<N$$ 均取不同值保证的。

该算符的本征态为 

$$
   \vert u_s\rangle=\frac{1}{\sqrt{r}}\sum_{k=0}^{r-1}e^{-2\pi isk/r}\vert x^k\;\mod N\rangle. \; (0\leq s\leq r-1)
$$

对应本征值为 $$e^{2\pi i s/r}$$。由于 U 的所有本征态之等振幅和恰好为 $$\vert 1\rangle$$，因此我们可以进行相位估计(注意该相位估计的下半部分完全是经典线路)。态输入为 $$\vert 1\rangle$$，算符输入为 $$U_s$$，操作时，对于 $$U^{2^j}$$ 可以迭代计算，确保不需要进行指数次 U 作用。相位估计的最终结果测量值为 $$s/r$$ 的若干位二进制小数表示。通过经典的连分数算法，我们可以高效的确定 r 的正数值。注意这里需要 $$t>2n+1$$ 位，其中 $$2^n>N$$，以保证连分数估计的精度。

如果 s 和 r 有共同因子，则测量结果可以约分，导致连分数算法无法直接得到 r。这时的最佳实践，是运行相位估计两次，得到 $$s_1'/r_1', s_2'/r_2'$$，那么真实的 r 将是 $$r_1',r_2'$$ 的公倍数。

解决阶数问题，找到 r 之后，我们来看如何最后分解 N。我们需要两个定理。

定理1: N 是 L bit 的合数，$$x^2=1\;\mod N$$ 且 $$x\neq \pm 1\;\mod N$$。则 $$\gcd(x-1,N)$$ 和 $$\gcd(x+1,N)$$ 中至少有一个 N 的因子。

定理1的理解比较简单，只需要考虑 $$(x+1)(x-1)=kN$$ 即可。

定理2: 设 $$N=\prod^m p_i^{\alpha_i}$$ 是一个具有全奇素数分解的整数。设 $$1\leq x\leq N-1$$ 且 x 与 N 互质，r 是 x 关于N 的阶数，那么 r 是偶数且 $$x^{r/2}\neq -1\;\mod N$$ 的概率大于 $$1-1/2^m$$。

相应的合数分解算法为：

   1. 若 N 为偶数，直接返回 2.
   2. 判断是否存在 a,b，使得 $$a^b=N$$，若存在，返回 a。
   3. 随机选取 $$1\leq x\leq N-1$$，若 $$\gcd(x,N)>1$$，返回该公约数。
   4. 使用量子线路，计算 x 关于 N 的阶数 r。
   5. 若 r 为偶数，且 $$x^{r/2}\neq -1\;\mod N$$，进入6，否则重新进行3。
   6. 计算  $$\gcd(x^{r/2}-1,N)$$ 和 $$\gcd(x^{r/2}+1,N)$$ ，检查其中哪个是 N 的因子，并返回该公约数。

算法的一些细节评述：

   * 所有的求公约数 gcd 的操作，都需要用辗转相除法，用来保证多项式时间。
   * 第二步，需要依次尝试 $$N^{1/k}$$，直到 $$k=\lceil \log_2N\rceil$$，所以也是多项式时间。
   * 只有第 4 步需要用到量子线路，其他都是经典算法。事实上，可以证明，合数分解和阶数寻找的复杂度多项式等价。（两问题可互相解决）
   * 之所以不需要判断 $$x^{r/2}=1\;\mod N$$，是因为 r 作为阶数的定义，是最小的使 x 指数余1的整数，因此 $$r/2$$ 指数不可能余1。
   * 算法保证了核心检查步骤 5，至少具有 $$3/4$$ 的成功概率。

## 7. Grover 算法

在大量态的叠加态中放大一个或若干个标记态的振幅，使得测量结果大概率处于这些标记态。标记态由黑箱 $$I_\tau=I-2\vert\tau\rangle\langle\tau\rangle$$ 给出，也即作用黑箱之后，标记态振幅反号，其他态振幅不变。对于经典算法，在N个无序数据找到标记态需要的时间为 $$O(N)$$，利用量子线路的 Grover 算法可以实现 $$O(\sqrt{N})$$ 复杂度。
1. 原始算法步骤
  
两个基本操作：反转标记态振幅的黑箱 $$I_\tau$$。反转非 $$\vert+\rangle$$ 态振幅的算符，构造为 $$I_{!{+}}=-H^nI_0H^n=-I+2\vert +\rangle\langle +\vert$$，其中 $$I_0=I-2\vert 0\rangle\langle 0\vert$$，$$\vert +\rangle=\frac{1}{2^{n/2}}\sum_{i=1}^{2^n}\vert i\rangle$$。

假设初态为 $$2^n$$ basis 的等幅叠加，也即 $$\vert +\rangle$$，交替作用 $$I_\tau, I_{!+}$$。相当于波函数在 $$\vert +\rangle$$ 和 $$\vert \tau\rangle$$ 确定的平面里做镜面反射(镜面分别为$$\vert \tau^\perp\rangle,\vert +\rangle$$)，注意被两个算符反复作用后，波函数总是能表示成 $$\vert\psi\rangle=c_1\vert \tau\rangle+c_2\vert +\rangle$$ 的形式。在该平面与 $$\vert +\rangle$$ 垂直的态我们记为 $$\vert +^\perp\rangle\propto -\langle +\vert \tau\rangle\vert +\rangle+\vert \tau\rangle$$。由于最简单情形有 $$\langle +\vert\tau\rangle=\frac{1}{2^{n/2}}$$，可得标记态和 $$\vert +^\perp\rangle$$ 夹角近似为 $$\beta=1/\sqrt N$$. 

每作用两次镜面，等价于一个靠近 $$\tau$$ $$2\beta$$ 角度的旋转。本来离目标的角度是 $$\pi/2-\beta$$，那么 T 次旋转后，最接近目标，也即 $$(2T+1)\beta=\pi/2$$， 也即 $$T=\pi/(4\beta)-1/2\approx \pi\sqrt{N}/4$$。可见搜索次数获得平方加速。

对于有多个标记态的例子，以上分析完全同理，只不过此时夹角 $$\beta\approx \langle\tau\vert +\rangle=\sqrt{k/N}$$，那么最后需要的搜索步骤为 $$\pi/4 \sqrt{N/k}$$。

更进一步地，如果初态为 $$U\vert 0\rangle$$，搜索算法和 grover 完全相同，只需将 $$I$$ 替换为 $$I=-UI_{0}U^{-1}$$ 即可。这就是振幅放大算法。其需要的搜索次数为 $$\frac{\pi}{4A_{\tau 0}}$$。也就是说符合条件的态本来 $$A\vert 0\rangle$$ 中测量的概率是 $$p=A_{\tau0}^2$$，也就是需要制备和测量 $$O(p)$$ 次才能找到符合条件的态。但经过振幅放大，只需 $$\sqrt{p}$$ 次放大，就可以以 $$O(1)$$ 的概率找到相应态。

2. 相位匹配

对于 Grover 算法中出现的两个镜面反转操作，我们还可以加上一个相角，类似旋转操作，也即 $$I^\theta_{\vert x\rangle}=I+(e^{i\theta}-1)\vert x\rangle\langle x\vert$$. 当 $$\theta=\pi$$ 时，我们回到原始的搜索算法。相应的搜索算法可以改造为利用算符 $$I_{\tau}^\theta$$ 和 $$I_{0}^\phi$$。此时，两个相角并不是任意取都可以用来逼近目标态。只有满足所谓的相位匹配条件时，才可以成功逼近标记态。该条件为 $$\theta=\phi$$。注意该相位匹配条件默认了数据库制备 $$U\vert 0\rangle$$ 和搜索引擎 $$U'I_0^\theta U'^{-1}$$ 使用的矩阵相同，即 $$U=U'$$。

更广义的，如果我们把两个 U 区分开，则有更普适的相位匹配条件。

$$
\tan\frac{\theta}{2}\left[\cos 2\beta +\tan\theta_0\cos\delta\sin 2\beta\right]=\tan\frac{\phi}{2}\left[1-\tan\theta_0\sin\delta\sin 2\beta\tan\frac{\theta}{2}\right]
$$

其中的 $$\theta,\phi$$ 为上述的两个旋转相角。对于初态，$$\vert 1\rangle,\vert 2\rangle$$ 分别为 U 定义的初态在标记态和非标记态上的分量，也即 $$U\vert 0\rangle=\sin\beta \vert 1\rangle+\cos\beta \vert 2\rangle$$。但为了将搜索引擎的 U 和初态区分，我们取实际初态 $$\vert \psi_0\rangle=\sin\theta_0\vert 1\rangle+cos\theta_0e^{i\delta}\vert 2\rangle$$。

*quant-ph/0107013*

3. 龙算法

Grover 算法通过两次镜面使得量子态接近标记态，但是无法保证最优搜索次数时测量得到标记态的概率为 1。Long 算法进行了改进，利用上述的旋转代替镜面反转，通过调节旋转的 phase，可以使得指定搜索次数后标记态出现概率达到 1.

初始数据库为 $$\vert +\rangle$$ 态。同样地， $$\beta\approx 1/\sqrt N$$。由于相位匹配的要求，则两个旋转角度需要相同，记为 $$\phi$$。通过计算，可以发现实现零失败概率的相角为
$$
\phi=2\arcsin \left(\frac{\sin\frac{\pi}{4J+6}}{\sin\beta}\right).
$$
其中 $$J\geq J_{op}$$ 是需要的搜索次数减1，我们在 J+1 次搜索后测量。之所以次数不小于 Grover 算法的次数，是因为 $$\arcsin$$ 自变量需要不大于 1。

*PhysRevA.64.022307*

## 8. Steane code

Steane code 由 7 个物理比特编码一个逻辑比特。只考虑单比特错误，则有 $$3*7+1=22$$ 种情况。其 stablizer 生成元为 $$M_1=X_4X_5X_6X_7, M_2=X_2X_3X_6X_7, M_3=X_1X_3X_5X_7$$, $$N_1=Z_4Z_5Z_6Z_7, N_2=Z_2Z_3Z_6Z_7, N_3=Z_1Z_3Z_5Z_7$$. 对应的逻辑态为 

$$
   \vert 0_L\rangle=\frac{1}{2\sqrt{2}}(1+M_1)(1+M_2)(1+M_3)\vert 0000000\rangle\\=\frac{1}{2\sqrt{2}}(\vert 0000000\rangle+\vert 0001111\rangle+\vert 0110011\rangle+\vert 1010101\rangle+\vert 0111100\rangle+\vert 1100110\rangle +\vert 1011010\rangle+\vert 1101001\rangle).
$$

$$
\vert 1_L\rangle=\frac{1}{2\sqrt{2}}(1+M_1)(1+M_2)(1+M_3)\vert 1111111\rangle\\=\frac{1}{2\sqrt{2}}(\vert 1111111\rangle+\vert 1110000\rangle +\vert 1001100\rangle +\vert 0101010\rangle+\vert 1000011\rangle +\vert 0011001\rangle+\vert 0100101\rangle+\vert 0010110\rangle).
$$

注意到对于 n 个物理比特的编码方案，其 stablizer 或者说 error syndrome 必然是 n-1 个。这也就对应了 $$2^n$$ 维  Hilbert space 中存在 $$2^{n-1}$$ 个限制，这确保了剩余的空间恰好为 2 维，对应一个 logical qubit。

## 9. 5 qubit code

事实上，我们最少只需要5个物理比特来保护一个逻辑比特，并且可以纠正任意单比特错误。

其4个 stablize 为 $$M_1=IXZZX$$,  $$M_2=XIXZZ$$, $$M_3=ZXIXZ$$, $$M_4=XZZXI$$。事实上，其都是 XIXZZ  字符串的轮换。根据对称性我们还有 $$M_5=ZZXIX$$，但是容易看出 $$M_5=M_1M_2M_3M_4$$，因此并不独立。注意这里的算符表示与 8 稍微不同，用 8 的语言则为 $$M_1=X_2Z_3Z_4X_5$$。

同样地，我们有逻辑态 $$\vert 0_L\rangle=\frac{1}{4}(1+M_1)(1+M_2)(1+M_3)(1+M_4)\vert 00000\rangle $$, $$\vert 1_L\rangle=\frac{1}{4}(1+M_1)(1+M_2)(1+M_3)(1+M_4)\vert 11111\rangle $$.

展开可写成 

$$
0_L\sim 00000+10010+01001+10100+01010+00101-10001-01100-00110-11000-00011-11101-11011-11110-01111-10111
$$

同理，逻辑 1 态只需将上述各分量 01 反转。
其态线性组合的规律为，0 态只包含偶数个 1，将 5 位看作周期性闭环，则没有 1 相邻时为正号，否则贡献 -1 的 phase。更具体的，16 个基分别为 00000， 10100 共 5 个轮换，11000 共 5 个轮换（负），11110 共 5 个轮换 （负）。

## 10. CSS code

以 Steane 码为例说明 CSS 码的构造思路。 CSS 码是一类基于经典纠错码构造量子编码的通用框架。其主要运用两组经典线性码来纠正量子错误。

假设 $$C_1,C_2$$ 为两组线性码空间 $$[n, k_1]$$, $$[n, k_2]$$，也即是用 n 个字符编码 $$k_i$$ 个字符。且 $$C_2\subset C_1$$。如果 $$C_1$$ 和 $$C_2^\perp$$ 都是纠正 t bit 错误的经典码，那么可用 $$C_1,C_2$$ 构造出可以纠正 t bit 量子错误的量子码。该量子码可用 n bit 来保护 $$k_1-k_2$$ bit。

进一步概念解释：线性码只是一个由部分 01 字符串长成的空间而已，也就是说我们不要求 $$C_2$$ 有经典纠错能力。但 $$C_2^\perp$$ 是需要经典纠错的编码。这里的$$\perp$$ 意义是，$$G'=H^T, H'=G^T$$，其中 G 是生成矩阵，H 是检查矩阵。需要注意 $$C_2^\perp\subset C_2$$ 是可能的，因此这里不能简单的按照欧氏空间的正交来理解。 

CSS code 的逻辑比特组成为

$$
  \vert x+C_2\rangle\equiv \frac{1}{\sqrt{C_2}}\sum_{y\in C_2}\vert x+y\rangle.
$$

注意到，对于任意 $$x\in C_1$$，存在 $$\vert C_2\vert$$ 个符合以上定义的态对应同一个态。因此独立的逻辑态的数目为 $$\vert C_1\vert/\vert C_2\vert=2^{k_1-k_2}$$，也即逻辑比特定义在二者的商群上。也即该 CSS code 用 n 比特保护了 $$k_1-k_2$$ 比特，并可以纠正至多 t 比特的错误。

下面是结合 Steane code 的例子和具体理解。

Steane 码是 CSS code 的一个具体实现，其需要的经典码 $$C_1$$ 的 H 矩阵为

$$
  H=\begin{pmatrix}0&0&0&1&1&1&1\\0&1&1&0&0&1&1\\1&0&1&0&1&0&1\end{pmatrix}.
$$

我们取 $$C_2=C_1^\perp$$，因为 $$k_1=4, k_2=3$$，那么构成的 Steane code 可以纠正至多一个比特的错误。对应 

$$
  H(C_2)=G(C_1)^\perp=\begin{pmatrix}I_{4\times 4}&\begin{matrix}0&1&1\\1&0&1\\1&1&0\\1&1&1\end{matrix}\end{pmatrix}.
$$

我们还需要进一步证明 $$C_2\in C_1$$ (因为 $$H(C_1)H(C_1)^\perp=0$$)。 另一方面，因为 $$C_2^\perp=C_1$$ 符合可以纠错 $$t=1$$ 比特的条件。因此利用 $$C_1,C_2$$ 两个经典线性码空间，可以实现 7 比特编码1比特并至多纠错 1 比特的 CSS code，也即 Steane code。

根据 CSS code 逻辑态的一般公式，可以得到 Steane code 的逻辑态组成：

$$
\vert 0_L\rangle=\frac{1}{8}(\vert 0000000\rangle+\vert 0001111\rangle+\vert 0110011\rangle+\vert 1010101\rangle+\vert 0111100\rangle+\vert 1100110\rangle +\vert 1011010\rangle+\vert 1101001\rangle).
$$

逻辑 1 态同理，只需反转 0 态的 10 即可。


## 11. 对偶量子计算

思想：可以计算 Unitary 的和，而不只是传统量子线路中的 Unitary 的积。可以通过量子干涉来实现这一点。

步骤：物理实现，类似双缝干涉的装置，在不同的小孔，做不同的 Unitary 操作。则最后屏上的态为 $$\sqrt{p_1}U_1+\sqrt{p_2}U_2$$，实现了 linear combination of unitaries (LCU)。

量子线路实现：![](https://user-images.githubusercontent.com/35157286/59549591-99cf2f00-8f92-11e9-9460-bbad4ac94937.png)

以上线路，在辅助位测量结果为 0 时，对应上方线路的输出为 $$(U_0+U_1)\vert \phi\rangle$$。如测量结果为 1，此时对应 $$U_0-U_1$$，则从头重新开始。

## 12. 相位估计算法

回忆量子傅立叶变换，定义为
    
$$
    \vert j\rangle \rightarrow\frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}e^{2\pi i j k/N}\vert k\rangle.
$$

将量子态写作二进制数的直积态的形式，则有：
    
$$
    \vert j_1...j_n\rangle\rightarrow\frac{1}{2^{n/2}}\prod_{i=1}^n e^{k_i\times 0.j_{n+1-i}...j_n}\vert k_i\rangle \\=\frac{(\vert 0\rangle+e^{2\pi i 0.j_n}\vert 1\rangle)...(\vert 0\rangle+e^{2\pi i 0.j_1j_2...j_n}\vert 1\rangle)}{2^{n/2}}.
$$

其中 $$2^n=N$$，n 为 qubit 数目，$$0.j_1…j_n=j_1/2^1+…j_n/2^n$$. 实现傅立叶变换的线路如下。

![](https://user-images.githubusercontent.com/35157286/47368082-8e817780-d713-11e8-8359-451b7bbdcee3.png)
其中 $$R_n=Diag[1, e^{\pi i/2^n}]$$. 注意线路图最后省略了一个交换线路，生成的 y 的顺序正好是相反的。

现在我们来看相位估计算法，其用来估计本征值问题。具体地，已知 Unitary matrix $$U$$ 和其本征态 $$u$$，则 $$U u=e^{2\pi i \phi}u$$，该算法用来估算 $$\phi$$。该算法分为两部分，第一部分线路如下图。

![](https://user-images.githubusercontent.com/35157286/59552297-ab76fd80-8fb7-11e9-90e8-879af40b2650.png)

用紧凑的写法，上述算法为
    
$$
    \vert 0\rangle\vert u\rangle\rightarrow \sum_{j=0}^{2^t}e^{2\pi i \phi j}\vert j\rangle.
$$

注意到其末态恰好类似傅立叶变换的输出态，只需将傅立叶变换的量子线路取共轭，也即逆傅立叶变换的线路，作为相位估计的第二部分，那么输出的前 t bit 的测量结果，恰好可以看作 $$\phi$$ 的二进制小数近似，也即输出近似为 $$0.t_1t_2..t_t$$. 相位估计算法整体的线路为。

![](https://user-images.githubusercontent.com/35157286/59552380-c007c580-8fb8-11e9-880d-4555e4d77b89.png)

若想以 $$1-\epsilon$$ 的概率实现 n 位 $$\phi$$ 的准确近似，则需要 $$t=n+\lceil\log( 2+\frac{1}{2\epsilon})\rceil$$.

如果我们输入的态 $$\vert u\rangle$$，不是 U 的本征态，那么根据量子力学的线性叠加原理，其输出态测得的值还是对应本征值的近似 $$\phi_i$$，测的概率为 $$\vert c_i\vert^2=\langle \phi_i\vert u\rangle$$。这为我们进一步利用相位估计奠定了基础。