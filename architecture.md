# Architecture

## Revised draft

gloop は device loop と host loop という2つの event loop, そして system 全体を調停する scheduler を提供し, shared GPUs における GPU 利用の調停を行う. event loop は (1) user 透過な GPU kernel の suspend と resume のための scheduling point の提供と (2) non-blocking な I/O operations の提供を行う.

![gloop](fig/multiple.png?raw=true)

図は gloop の architecture を示す. それぞれの application は 2種類の event loop を持つ. host 側に host loop を持ち, GPU 側では ThreadBlock ごとに device loop を持つ. scheduler はそれぞれの host loop を束ね, system 全体での GPU の利用を調停している.

application は host loop を介して GPU kernel を実行する. host loop は scheduler に問い合わせ, GPU kernel の実行権を取得し GPU kernel を実行する. GPU kernel 内では device loop は event loop abstraction が提供し, またこれを用いた gloop APIs を提供する. gloop APIs は non-blocking な I/O operations であり device-host 間 IPC を通じて I/O operation を発行することができる. I/O operation が完了すると device loop は callback を呼び出す.

host loop は IPC を経由して device loop からの I/O request を受け取り, asynchronous に I/O を発行する. そして I/O が完了した場合, device loop に向かって IPC で通知する. このため, device loop を抜けることなく, I/O を実行することができる. GPU kernel code を記述するだけで host I/O も含んだプログラムを記述することができる.

scheduler はそれぞれの host loop からの実行権 request 及び実行時間を監視しており, (1) 実行中以外の host loop からの実行権 request があり, (2) 実行中の host loop が一定以上の時間 GPU kernel を実行し続けている時, 実行中の host loop 及び corresponding device loop に GPU kernel の suspend を行うよう request する. device loop は event loop の task 消費のタイミングで実行状態, すなわち queued tasks を含んだ device loop の状態を保存し GPU kernel を終了させる. これによって他の host loop に GPU kernel 実行権を明け渡すことができる.
host loop は GPU kernel 終了後, pending tasks が残っているかどうかを確認し, 残っていた場合, 自動的に再び kernel 実行権を取得し GPU kernel を実行を試みる. この際, device loop の状態は以前保存されたものから resume される. gloop APIs user 側からは透過的に GPU kernel が suspend, resume される.

device loop は他の host loop による kernel 実行権 request が存在しない場合中断しない. GPU kernel launch は costly operation である[GPUnet]. gloop は必要があって初めて device loop を抜けることによって必要のない GPU kernel launch による latency を削減する.

malicious な application や gloop APIs を用いない application が存在した場合は, GPU kernel を kill する. これは GPU が non-preemptive であるため, kernel の実行の中断を gloop のような support なしに行うことができないためである.


## Discussion

## Draft

![gloop](fig/multiple.png?raw=true)
*Architecture of gloop*

_gloop_ は device / host の両側にまたがる runtime を提供する.
GPU kernel が実行されると, なかで async I/O API, `gloop::API` を呼び出し, その引数に lambda や関数ポインタを与える.
_gloop::API_ はその lambda を保存し callback を registration する.
そして device-host 間 RPC (based on device-host shared memory) を利用して, host gloop に async I/O の発行を依頼する.
host gloop には async I/O event loop (_host loop_) が存在し, 実際に async I/O を発行する.
この時, device 側は2つの選択肢を持つ. (1) そのまま GPU kernel を抜けてしまう場合, async I/O が完了した後,
host 側から保存しておいた callback を再度 GPU kernel として発行する. この時, host に一度戻るため kernel invocation cost がかかってしまうため,
latency は大きくなる. 一方 device での GPU kernel を終了させるので, GPU の計算資源を無駄にせず, 別の GPU context が kernel を実行することができる.
(2) もうひとつは device 側で polling するもの. 図中の device loop がこれに当たる. host loop は async I/O が完了すると device loop に RPC で通知を行う.
device loop はこれを polling し, 完了を確認すると callback とした GPU kernel を発行する.
一度 host に戻らないため, latency は極めて低くなる余地がある一方, 他の context は GPU kernel を実行することができない.

この2つのタイプの callback の発行を, system 全体を見ながら gloop がスケジュールする. 例えば, system 全体で一つしかアプリケーションがない,
もしくはアプリケーションの budget が余っている場合は, device loop を用いて良い. 一方で budget が尽きる, shared GPU であるといった場合には,
device-host RPC を通じて device loop を止め, host loop driven で callback を発行することができる.
この2つの event loop, ２つの callback 発行タイプを用いて, latency を下げつつ, shared GPU aware な async I/O 機構を GPU に提供する.

async I/O の種類は多岐にわたる. FileSystem APIs はその代表例だが, Network I/O APIs もこれに含まれる.
更には, GPU はデータ転送とコンピュテーションをオーバーラップ & 非同期で発行できることを利用すると,
この async I/O にはデータ転送 (例えば, host-device memory copy, `gloop::MemcpyDtoH(..., ..., [] () { })` etc.) も含まれる.
データ転送も gloop API で発行することによって, データ転送を含めたプログラムがすべて GPU kernel のみで完結し,
パイプラインを構築するといったことを考慮する必要なく, 自然とデータ転送とコンピュテーションがオーバーラップし,
コンピュテーションリソースを使い切ることが可能である.

![GPUfs](fig/gpufs.png?raw=true)
*Architecture of gpufs*

## Memo

+ gloop に従わない / 使わない application が GPU を専有するのでは?
    + 専有する. ため, shared GPU の場合は, platform 側が app を kill することで対応する. 前提として長時間専有する app は kill される.
    + kill してしまうというのは, preemption がない以上現実的な解決策であり, disengaged scheduling も長時間動き続けるものに対してはそれを想定している. GPUvm でもそう言った.
    + preemption があれば, それを使うというのが手段となる
+ 同じ context から複数の別の kernel が同時に動かされた場合, 同時に GPU を empty にするのは難しいのでは?
    + 難しい, ため, host loop と連携して適当なタイミングで全部一旦止めるということをしてあげなければならない. gang scheduling のようなものを想定している.
    + これを scheduling してあげるというのも, ひとつの challenge.
    + しかし, Chimera や ISCA '14 の paper を引いて, 「将来的に SM 単位で別の context が動かせるなら, わざわざ全部 empty にする必要はなくって」というように言うこともできる.
+ mmap, msync などの API, gloop が host に抜けられるのであれば抜けて cudaHostRegister で本当に mmaped memory を与えれば良い
    + GPU kernel 実行中は GPU page table を変えられない, のであれば, 一回抜けて戻れば良い
