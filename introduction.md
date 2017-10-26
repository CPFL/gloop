# Introduction

## Draft

GPU は graphics 目的を越え massively data-parallel computing での利用に用いられている.
general purpose computing on GPU (GPGPU) の用途は拡大し,
科学技術計算のみならず, database, や web application frontend,
high speed packet processing, ssl reverse-proxy といった用途にも用いられる.

このような汎目的の利用は, GPU の計算資源としての抽象化が進むに連れてより活発になるであろう.
GPU 仮想化技術の発展によって, GPU は共有されるリソースとしてクラウド環境で提供することが可能になっている.
また仮想 GPU のみならず, コンテナ仮想化の隆盛によって,
GPU を共有リソースとしてコンテナ間で使うことも容易になっている.
GPU を共有リソースとして管理することによって, 利用率を向上させ,
GPU を (言い方考える) 切り売りすることができ,計算資源として抽象化することができる.
GPU の計算資源としてクラウド環境への統合は，
クラウド上で動作するサーバサイド・アプリケーションに GPU を用いることを可能とする.

サーバサイド・アプリケーションといった GPU の新たな利用法によって,
GPU のカーネルを動かし続ける新しい class のアプリケーションがあらわれている.
(
    CPU - GPU のデータ転送とコンピュテーションはオーバーラップできる.
    最大限利用しようとすればパイプラインを自分で構築する必要があり, これはプログラミングを複雑にする.
    これを隠蔽するようなプログラミングモデルとして, GPUfs は File System という, prefetching,
    caching を持った abstraction を GPU 側に提供し, パイプラインを構築せずにオーバーラップさせることができることを志向した.

    また, データが CPU 側にあることは, input / output の境界で CPU 側に GPU コードが戻る必要があることを意味する.
    この場合, CPU 側からまた GPU kernel を invoke するため, 低いレイテンシを実現することが困難である.
    GPUnet はカーネルを動かし続けることによって, GPU kernel の invoke のコストを避け, 低いレイテンシを達成した.
    また GPU の中から networking APIs を発行することによって, input / output を GPU の計算途中に順次発行することができた.
    これは input / output におけるレイテンシを削減した.
    (CPU 側に戻った時にしか I/O を扱えないため, 計算途中の結果を順次送るといったことができず, 全部計算してから送るということしかできなかった)
)
このようなアプリケーションでは GPU カーネルは CPU と RPC によって interaction を行い,
CPU 側のみが実行可能な処理, 例えば I/O を発行する.
これによって, GPU - CPU での interaction によってプログラムが複雑になるのを避け,
GPU 側コードのみで完結するようなプログラミングモデルを提供する.
また, カーネルの実行途中において I/O を発行することができるほか,
CPU 側から GPU カーネルの invoke を毎回行わないため,
より低いレイテンシで GPU で処理を行うことができる.

```
4つめ5つめを再構成して, ここに具体的な例, 例えば GPUnet ならこう, GPUfs ならこうというふうな文章を入れる.
For example... という形. 基本的にはパラグラフの末尾に入れる.
```

しかしながら, このようなアプリケーションは GPU を専有してしまう.
GPU カーネルは preepmtion を行うことができないので,
GPU カーネルが persistent に走り続けるアプリケーションは GPU を専有し,
アプリケーションが動作している間は他のアプリケーションが GPU を利用することはできない.
また, preemption が可能になったとしても, GPU は巨大な register file を持つため context switch は非常に時間がかかってしまうため[ISCA'14, ASPLOS'15],
低いレイテンシを要求するアプリケーションは実行することができない.

GPU を専有してしまうアプリケーションは, GPU を共有するクラウド環境上で動作させることができない.
クラウド環境が GPU を利用率向上のために他の user と共有した場合, user が GPU を専有することはできない.
pass-through を用いた場合 GPU を専有することができるが, GPU の利用率の低下し,
また GPU を専有するため user にとってのコストが高い.
既存の手法ではこのような場合, user を kill するといった手法が取られてきた.
このため, GPU を共有するクラウド環境上で低レイテンシを志向する GPU を用いたアプリケーションを実行することはできない.

(
    ここに, この問題はそもそもなんで起こるのかを端的に示す paragraph を入れる.
    想定としては, resource manager が I/O と non-preemptive nature を統合した
    scheduling 抽象を持たないからだというふうに言えるのではないか?
    そのような抽象を持たないから, user が自前で polling や RPC をするはめになっているのでは?
    (half-baked idea)
    GPU がプリエンプションできないデバイスであるから,
    イベントポンプのような抽象を resource manager が提供すべきなのに, 提供できていない.
    poor resource management.
)

本提案では, GPU のプログラミング interface に新たな抽象を導入する.
共有 GPU の resource manager は新たにイベントループの抽象を導入し,
これをユーザの GPU カーネルに対して公開する.
イベントループを用いた非同期 I/O の interface を GPU カーネルに公開することによって,
タスクの scheduling を resource manager が行うことができるようにする.
GPU カーネルにおける user レベルでの polling をなくし,
共有 GPU において GPU が専有されない状態を維持しつつ,
カーネルを動かし続ける利点,
低いレイテンシや GPU カーネルで完結する simple なプログラミング・モデルを維持する.

scheduling を行いつつ低いレイテンシを維持することは challenge である.
我々はこれを解決するために, XXX (今の想定, adaptive polling) を導入する.
XXX.

YYY 別のchallengeについて記述.
(compiler レイヤでの解決についてだと思う, 関数ポインタなどなくうまくやるには?)

(実験について触れる.)
ZZZ

## Comparison


|   | GPUfs | GPUnet | PTask | Rhythm | current |
| :-: | :-: | :-: | :-: | :-: | :-: |
| I/O from GPU                           | * | * |   |   | * |
| Multiple Application                   |   |   | * | * | * |
| Low Latency                            |   | * |   |   | * |
| High Throughput                        | * | * | * | * | * |
| Modularity                             |   |   | * |   |   |
| Ease of Programming (OR ちらばらない)  | * | * |   |   | * |
| Shared GPU aware                       |   |   | * |   | * |


## Comments

+ preemption がでたらどうするか? preemption がすべてを解決するのでは?
    + preemption はコストが高いから, すべて preemption でさせるよりも, user が明示的に「非同期タスク待ちです」と伝えるところで切る方が効率的. preemption ができれば, 不当に長時間 GPU を利用する user への最終手段としての preemption による stop が可能になる. このため, preemption と本提案は直行し, 本提案は preemption を用いてより効率的な管理を行うことができる.
    + Chimera の tradeoff の話とかをいれる
+ clear にすべきなこと
    + PTask ではこれができない, GPUnet はこれができない, GPUfs はこれができないみたいな星取表がほしい
    + 特に直接に競合するのは PTask なので, これとの差分をはっきりさせるべき
        + latency sensitive な data center application については PTask は(当時それほど隆盛になかったから) 考えていない
        + latency については考えているが, GPUnet のように polling するというレベルまで latency を severe には考えていない, 評価結果も, latency 9ms など. (GPU kernel invocation の latency は 25us)
            + これについてはもう少し考える必要がある
            + 基本的に PTask は channel が queue を持っていて, 中に task がある & GPU が available なら kernele を invoke, なので, GPUnet の kernel での polling のような考え, 及びこれが必要とするような latency 制約までは持っていない
        + multi user については考えていない

## Memo

+ p2 くらいに PTask の話をうまく入れる
+ 低レイテンシ, 高スループットを志向するところまで持っていくことができれば, datacenter application の話の流れにのせることができる (IX とか)
+ resource management のレイヤでやらずとも, CPU 側のコードで dispatch などの polling loop をすればよいのでは.
    + ただし, CPU 側でやる場合は, ローカルな情報しかないので, 全体でスケジューリングするというようなことはできない
+ GPU のイベントループサービス
+ persistent threads とかの場合は, thread 抽象を肩代わりする & GPU task の実行時間を短くできる
    + shared GPU のようにすると長時間 task を動かすと殺される
+ GPUnet を複数動かすことができる
    + GPUnet のアプリケーションローカルで dispatch を CPU でやると, 常に CPU に戻る必要がある
    + shared GPU 全体で GPU 利用者が一人しかいなければ, GPU でループを回せば良い
+ 今回は NVIDIA driver / NVIDIA runtime をそのままつかう
    + これも利点の一つ
+ scheduling はどうなのか?
    + user はまったく関係ないので, fairness はきちんと管理できないとダメ
    + resource reservation みたいのもできないとダメ
    + scheduling については latency 保証みたいなのはできるのか?
        + 要検討
+ どの程度アプリケーションを書き換える必要があるのか?
    + GPUfs / GPUnet のようなアプリケーションの場合, sequential だったものを callback based な async I/O に書き換える必要がある
    + 他のものについては, host で I/O してというモデルからの書き換えが必要
        + 最終的には GPU 側コードだけになるので, logic は simplify される
+ interface の使い方を間違えても他の apps は大丈夫?
    + 影響は受けない. ただし, polling のようなことをすると, 他の GPU Apps は影響を受ける
    + このケースについては Architecture の方で対処法を語っている, 内容としては, kill すれば解決するというスタンス
+ "「GPU のカーネルを動かし続ける新しいクラスのアプリケーション」っていうのは何か名前はないの？"
    + "他の論文で使われている名前があったらその名前を使いましょう. もし名前がないなら, 名前を定義したほうが主張がはっきりしてよさそうです"
    + I/O を発行するようなものは GPUfs / GPUnet, こちらの方には特に特定の名前はなし.
    + computation を consumer / producer に分けて発行し続けるものは, persistent threads と呼ばれる.
