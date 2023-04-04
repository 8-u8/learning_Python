一般化線彂混合モデルに正則化項を追加したモデルを定式化するには、以下のように記述することができます。

・一般化線形混合モデル(GLMM)

$$y_i = x_i\beta + z_i\alpha + \epsilon_i$$

・L1正則化項を追加したGLMM

$$y_i = x_i\beta + z_i\alpha + \epsilon_i + \lambda \sum_{j=1}^{p}|\beta_j|$$

・L2正則化項を追加したGLMM

$$y_i = x_i\beta + z_i\alpha + \epsilon_i + \lambda \sum_{j=1}^{p}\beta_j^2$$

ここで、$y_i$は観測値、$x_i$は説明変数、$\beta$は説明変数のパラメーター、$z_i$はランダム効果、$\alpha$はランダム効果のパラメーター、$\epsilon_i$は誤差項を表し、$p$は説明変数の数、$\lambda$は正則化項のパラメーターを表します。

L1正則化項は、絶対値を使用し、L2正則化項は二乗を使用します。

これにより、パラメーターが大きすぎないように制限することで、過学習を防ぐことができます。

$$L(\beta, \sigma^2, \theta)= -\frac{n}{2}log(\sigma^2) -\frac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta) -\frac{1}{2}log|K|-\frac{1}{2}u^TKu + \lambda \beta^T\beta$$
