# METR-LA adjacency source

`adj_mx.pkl` was downloaded from
`https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl`
on 10 October 2026. SHA-256:
`a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3`.
The DCRNN source repository carries an MIT license (Yaguang Li, 2017);
its full license text is included as `DCRNN_LICENSE`.

The pickle holds ordered sensor IDs, an ID-to-index mapping, and a 207×207
weighted adjacency. Its sensor order matches `METR-LA.csv` exactly. Under
the experiment's `>0.1` threshold it has 1,515 non-self directed edges and
207 self edges. As with any pickle, load this file only from its verified
source and check its hash before use.
