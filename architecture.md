# Architecture

## Draft

![Architecture Figure](fig/architecture.png?raw=true)

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
