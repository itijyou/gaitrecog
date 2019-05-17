from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
torch.backends.cudnn.benchmark = True

from itertools import chain
from numbers import Integral

class ImageListEx(ImageList):
    "`ItemList` for computer vision extended."
    def __init__(self, *args, open_mode='file', **kwargs):
        super().__init__(*args, **kwargs)
        def _array2image(x):
            x = pil2tensor(x,np.float32)
            return Image(x)
        self.open_func = {'array':lambda x:_array2image(x),'file':self.open}[open_mode]
    def get(self, i):
        res = self.open_func(super(ImageList, self).get(i))
        self.sizes[i] = res.size
        return res

class ItemListsEx(ItemLists):
    "`ItemList` for each of `train` and `valid` (optional `test`) extended."
    def __getattr__(self, k):
        ft = getattr(self.train, k)
        if not isinstance(ft, Callable): return ft
        fv = getattr(self.valid, k)
        assert isinstance(fv, Callable)
        def _inner(*args, **kwargs):
            self.train = ft(*args, from_item_lists=True, x=self.train, **kwargs)
            assert isinstance(self.train, LabelList)
            kwargs['label_cls'] = self.train.y.__class__
            self.valid = fv(*args, from_item_lists=True, x=self.valid, **kwargs)
            self.__class__ = LabelLists
            self.process()
            return self
        return _inner

# like npr.choice and npr.randint
_choice = lambda xs: xs[torch.randint(len(xs), (1,)).item()]
_randint = lambda x: torch.randint(x, (1,)).item()
class PairList(ImageList):
    "`PairList` for verification."
    def __init__(self, items1:ImageList, items2:ImageList=None, perm_len:int=0, **kwargs):
        kwargs['items'] = []
        super().__init__(**kwargs)
        self.items1 = items1 # gallery
        self.items2 = items2 or self.items1
        self.perm_len,self.rand = perm_len,perm_len>0
        if self.rand: self.items = array([None]*perm_len)
        else:
            gallery,probes = len(self.items1),len(self.items2)
            assert np.all([x < np.iinfo(np.uint16).max for x in (gallery,probes)])
            ia = np.tile(np.arange(gallery), probes).astype(np.uint16)
            ib = np.repeat(np.arange(probes), gallery).astype(np.uint16)
            pos = self.items1.inner_df.pid.loc[ia].values == self.items2.inner_df.pid.loc[ib].values
            self.items = [(a,b,p) for a,b,p in zip(ia,ib,pos)]
        self.copy_new.extend(['items1','items2','perm_len'])
    def get(self, i):
        if self.rand:
            dfa,dfb = self.items1.inner_df,self.items2.inner_df
            # for gait recognition
            if not hasattr(self, 'pids'):
                self.pids = uniqueify(dfa.pid, sort=True)
            # gallery: any person's seq
            pa = _choice(self.pids)
            a = _choice(dfa.loc[dfa.pid==pa].index.values.tolist())
            # probe: half pos and half neg
            pos = _randint(2)==1
            b = _choice(dfb.loc[np.logical_xor(dfb.pid==pa, not pos)].index.values.tolist())
            # sampled pairs
            self.items[i] = (a, b, pos)
        else: a,b = self.items[i][:2]
        return Image(torch.cat((self.items1[a].px, self.items2[b].px), 0))

class PairVerificationProcessor(CategoryProcessor):
    "`PreProcessor` that do nothing to items for preprocessing."
    def process(self, ds):
        if self.classes is None: self.create_classes(self.generate_classes(ds.items))
        ds.classes = self.classes
        ds.c2i = self.c2i

class PairVerificationList(CategoryList):
    "`ItemList` for pair verification."
    _processor=PairVerificationProcessor
    def __init__(self, items:Iterator, x:PairList, classes:Collection=[0,1], **kwargs):
        super().__init__(items, classes=classes, **kwargs)
        self.x = x
    def get(self, i):
        o = self.x.items[i]
        if o is None: return None
        o = int(o[-1])
        return Category(o, self.classes[o])

_flatten = lambda x: sum((list(i) for i in x), [])
def get_data(data_root, dataset, splits, bs, task, split):
    data_dir = data_root/dataset
    with open(data_dir/'data.pkl', 'rb') as f: data = pickle.load(f).astype(np.single)
    df = pd.read_csv(data_dir/'labels.csv', index_col=0)
    splits = [int(_) for _ in splits.split(',')]
    last_vl = max(df.loc[df.pid.between(splits[1]-1, splits[1]-0.5)].index.values)
    data -= data[:last_vl+1].mean(0, keepdims=True)
    if task=='tr':
        splits_l,splits_r = [0]+splits[:1],splits[:2]
        if split=='tv':
            splits_r[0] = splits_r[1]
            splits_l[1],splits_r[1] = splits[1],splits[2]
        else: assert split=='tr', f'Not defined to train on {split}'
    else:
        _splits = [0] + splits
        i = {'tr':0,'vl':1,'ts':2}[split]
        splits_l = [_splits[j] for j in (0,i)]
        splits_r = [_splits[j] for j in (1,i+1)]
    def _gen_subset(l, r):
        flag = df.pid.between(l, r-0.5)
        return data[flag],df.loc[flag].reset_index(drop=True)
    tr_x,tr_df,vl_x,vl_df = _flatten(_gen_subset(*x) for x in zip(splits_l,splits_r))
    tr_list = PairList(ImageListEx(tr_x, open_mode='array', inner_df=tr_df), perm_len=128*5000)
    if task=='tr':
        def _gen_vl_list(i):
            sm_inds = vl_df.reset_index().groupby(['pid','aid'],as_index=False).nth(i)['index']
            sm_vl_df = vl_df.loc[sm_inds].reset_index(drop=True)
            return ImageListEx(vl_x[sm_inds], open_mode='array', inner_df=sm_vl_df)
        vl_list_g,vl_list_p = [_gen_vl_list(i) for i in range(2)]
        vl_list = PairList(vl_list_g, vl_list_p)
    else:
        def _gen_vl_list(i):
            sm_inds = vl_df.gallery == i
            sm_vl_df = vl_df.loc[sm_inds].reset_index(drop=True)
            return ImageListEx(vl_x[sm_inds], open_mode='array', inner_df=sm_vl_df)
        vl_list_g,vl_list_p = [_gen_vl_list(i) for i in (1,0)]
        vl_list = PairList(vl_list_g, vl_list_p)
    tfms = rand_pad(0, (126, 86))
    ret = (ItemListsEx(data_dir, tr_list, vl_list)
           .label_const(label_cls=PairVerificationList)
           .transform((tfms, []))
           .databunch(bs=bs))
    return ret

def _calc_acc(preds, dfg, dfp):
    ga_list,pa_list = uniqueify(dfg.aid, sort=True),uniqueify(dfp.aid, sort=True)
    acc = -np.ones((len(pa_list),len(ga_list)), dtype=np.single)
    for i,pa in enumerate(pa_list):
        pflag = dfp.aid==pa
        for j,ga in enumerate(ga_list):
            gflag = dfg.aid==ga
            inds = preds[pflag][:,gflag].argmax(1)
            flag = dfp.loc[pflag].pid.array==dfg.loc[gflag].reset_index(drop=True).loc[inds].pid.array
            acc[i,j] = flag.sum() / max(1,len(flag))
    assert np.all(acc > -1)
    return acc
class RecorderEx(Recorder):
    "`Recorder` extended for gait recognition."
    def on_train_begin(self, pbar:PBar, metrics_names:Collection[str], **kwargs:Any)->None:
        self.add_metric_names(['recog_acc'])
        super().on_train_begin(pbar, metrics_names, **kwargs)
    def on_batch_end(self, train, num_batch, last_output, **kwargs:Any)->None:
        super().on_batch_end(train=train, num_batch=num_batch, last_output=last_output, **kwargs)
        if not train:
            dl = self.learn.data.valid_dl
            gallery,probes = len(dl.x.items1),len(dl.x.items2)
            # num_batch is invalid when not train
            if not hasattr(self, 'preds'):
                self.ibatch = 0
                self.preds = -np.ones((probes,gallery), dtype=np.single)
                self.acc = []
            elif self.ibatch == 0:
                self.preds[:] = -1
            inds = self.ibatch*dl.batch_size + np.arange(last_output.shape[0])
            self.preds[inds//gallery,inds%gallery] = last_output[:,1]
            self.ibatch = (self.ibatch+1) % len(dl)
    def on_epoch_end(self, epoch:int, num_batch:int, smooth_loss:Tensor,
                     last_metrics:MetricsList, **kwargs:Any)->bool:
        self.nb_batches.append(num_batch)
        if last_metrics is not None: self.val_losses.append(last_metrics[0])
        else: last_metrics = [] if self.no_val else [None]
        if len(last_metrics) > 1: self.metrics.append(last_metrics[1:])
        xl = self.learn.data.valid_dl.x
        self.acc.append(_calc_acc(self.preds,xl.items1.inner_df,xl.items2.inner_df))
        self.format_stats([epoch, smooth_loss] + last_metrics + [self.acc[-1].mean()])

@dataclass
class LearnerEx(Learner):
    def __getattr__(self, k):
        if k == 'recorder' and hasattr(self, 'recorder_ex'):
            return self.recorder_ex
        else: raise AttributeError

_init_w = partial(nn.init.normal_, mean=0., std=0.01)
_lrn = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2.)
_maxpool = nn.MaxPool2d(2, 2, 0)
_relu = nn.ReLU()
_dropout = nn.Dropout()
class LBNet(nn.Module):
    "Local @ Bottom."
    def __init__(self):
        super().__init__()
        self.zscore = batchnorm_2d(2, NormType.Batch)
        self.conv1 = conv2d(2, 16, 7, 1, 0, True, _init_w)
        self.conv2 = conv2d(16, 64, 7, 1, 0, True, _init_w)
        self.conv3 = conv2d(64, 256, 7, 1, 0, True, _init_w)
        self.fc = nn.Linear(256 * 21 * 11, 2)
    def forward(self, x):
        with torch.no_grad(): x = self.zscore(x)
        x = _relu(_maxpool(_lrn(self.conv1(x))))
        x = _relu(_maxpool(_lrn(self.conv2(x))))
        x = _dropout(self.conv3(x))
        return self.fc(x.view(x.size(0), -1))

import contextlib
@contextlib.contextmanager
def np_print_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

@call_parse
def main(
        gpu:Param("GPU to run on", str)=0,
        dataset:Param("Dataset to use", str)='casiab-nm',
        splits:Param("Ends of subsets (tr,vl,ts)", str)='50,74,124',
        model:Param("Model to use", str)='lb',
        opt:Param("Optimizer: 'sgd'", str)='sgd',
        lr:Param("Learning rate", float)=0.01,
        mom:Param("Momentum", float)=0.9,
        wd:Param("Weight decay", float)=0.0005,
        sched:Param("Learning rate schedule", str)='st-15-90',
        bs:Param("Batch size", int)=128,
        epochs:Param("Number of epochs", int)=240,
        task:Param("Task to do (tr/ts)", str)='tr',
        split:Param("Target split to use (tr/vl/tv/ts)", str)='tr',
        trained:Param("Load from trained model", str)=None,
    ):
    """Train models for cross-view gait recognition."""
    torch.cuda.set_device(int(gpu))
    data = get_data(Path('../data'), dataset, splits, bs, task, split)
    get_net = {'lb':LBNet, 'mt':None, 'gt':None}.get(model, None)
    if get_net: net = get_net()
    else: assert False, 'Not implemented for model {}.'.format(model)
    model_dir = Path('output')/dataset
    assert opt == 'sgd', f'Unknown opt method {opt}'
    opt_func = partial(optim.SGD, momentum=mom)
    learn = LearnerEx(data, net, opt_func=opt_func, metrics=accuracy, wd=wd, path=Path('..'), model_dir=model_dir)
    learn.callback_fns[0] = partial(RecorderEx, add_time=learn.add_time)
    if task=='tr':
        model_name = f'{dataset}_{model}_{opt}-{lr}-{mom}-{wd}_{sched}_bs{bs}_{split}'
        assert sched.startswith('st-')
        iters = array([float(x)*10000 for x in sched.split('-')[1:]])
        batches = len(data.train_dl) * epochs
        assert np.all(iters < batches)
        iters = np.append(iters, batches).astype(np.int)
        phs = [TrainingPhase(x).schedule_hp('lr', lr*0.1**i) for i,x in enumerate(iters)]
        learn.callback_fns += [
            partial(GeneralScheduler, phases=phs),
            partial(SaveModelCallback, every='epoch', name=model_name),
        ]
        learn.fit(epochs, 1)
    else:
        # callback_fns are never called in get_preds
        learn.callbacks += [learn.callback_fns[0](learn)]
        learn.load(trained)
        _ = learn.get_preds()
        xl = learn.data.valid_dl.x
        acc = _calc_acc(learn.recorder.preds, xl.items1.inner_df, xl.items2.inner_df)
        pacc = array([j[array(chain(range(i),range(i+1,acc.shape[1])))] for i,j in enumerate(acc)])
        print(pacc.mean())
        with np_print_options(formatter={'float':'{:1.7f}'.format}, threshold=sys.maxsize):
            print(pacc.mean(1))
            print(acc)
