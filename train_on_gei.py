from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
#from fastai.basic_train import load_learner
torch.backends.cudnn.benchmark = True
from numbers import Integral

class ImageListEx(ImageList):
    "`ItemList` for computer vision extended."
    def __init__(self, *args, open_mode='file', **kwargs):
        super().__init__(*args, **kwargs)
        def _array2image(x):
            x = pil2tensor(x,np.float32)
            x.div_(255)
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
    def __init__(self, items1:ImageList, items2:ImageList=None, perm_len:int=0, offset:int=0, **kwargs):
        super().__init__([], **kwargs)
        self.items1 = items1
        self.items2 = items2 or self.items1
        assert perm_len > 0, "Not implemented for perm_len <= 0 to cover all cases."
        self.items = array([None] * perm_len)
        self.offset = offset
        self.copy_new.extend(['items1', 'items2', 'offset'])

    def get(self, i):
        "This is only for training currently. Iterate on all cases for testing."
        assert isinstance(i, Integral)
        dfa,dfb = self.items1.inner_df,self.items2.inner_df
        # for gait recognition
        if not hasattr(self, 'pids'):
            self.pids = uniqueify(dfa['pid'], sort=True)
        # gallery: any person's seq
        pa = _choice(self.pids)
        a = _choice(dfa.loc[dfa['pid'] == pa].index.values.tolist()) - self.offset
        # probe: half pos and half neg
        pos = _randint(2)==1
        b = _choice(dfb.loc[np.logical_xor(dfb['pid'] == pa, not pos)].index.values.tolist()) - self.offset
        # sampled pairs
        self.items[i] = (a, b, pos)
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
def get_data(data_root, dataset, splits, use_vl, bs):
    data_dir = data_root/dataset
    with open(data_dir/'data.pkl', 'rb') as f: data = pickle.load(f).astype(np.single)
    df = pd.read_csv(data_dir/'labels.csv')
    splits = [int(_) for _ in splits.split(',')]
    last_tr,last_vl = [max(df.loc[df['pid'].between(x-1, x-0.5)].index.values) for x in splits[:2]]
    splits_l = [0] + splits
    if use_vl: splits[0] = splits[1]
    data -= data[:last_vl+1].mean(0, keepdims=1)
    def _gen_subset(l, r):
        if l >= r: return None,None
        flag = df['pid'].between(l, r-0.5)
        return data[flag],df.loc[flag]
    tr_x,tr_df,vl_x,vl_df,ts_x,ts_df = _flatten(_gen_subset(*x) for x in zip(splits_l, splits))
    tr_list = PairList(ImageListEx(tr_x, open_mode='array', inner_df=tr_df), perm_len=128*5000)
    vl_list = PairList(ImageListEx(vl_x, open_mode='array', inner_df=vl_df), perm_len=128*1000, offset=last_tr+1)
    tfms = rand_pad(0, (126, 86))
    return (ItemListsEx(data_dir, tr_list, vl_list)
            .label_const(label_cls=PairVerificationList)
            .transform((tfms, []))
            .databunch(bs=bs))

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

@call_parse
def main(
        gpu:Param("GPU to run on", str)=0,
        dataset:Param("Dataset to use", str)='casiab-nm',
        splits:Param("Ends of subsets (tr,vl,ts)", str)='50,74,124',
        use_vl:Param("If use vl for training", bool)=0,
        model:Param("Model to use", str)='lb',
        opt: Param("Optimizer: 'sgd'", str)='sgd',
        lr: Param("Learning rate", float)=0.01,
        mom: Param("Momentum", float)=0.9,
        wd: Param("Weight decay", float)=0.0005,
        bs: Param("Batch size", int)=128,
        epochs: Param("Number of epochs", int)=240,
    ):
    """Train models for cross-view gait recognition."""
    torch.cuda.set_device(int(gpu))
    data = get_data(Path('../data'), dataset, splits, use_vl, bs)
    get_net = {'lb':LBNet, 'mt':None, 'gt':None}.get(model, None)
    if get_net: net = get_net()
    else: assert False, 'Not implemented for model {}.'.format(model)
    model_dir = Path('output')/dataset
    learn = Learner(data, net, opt_func=optim.SGD, metrics=accuracy, wd=wd, path=Path('..'), model_dir=model_dir)
    batches = len(data.train_dl) * epochs
    ph1 = (TrainingPhase(batches*1/8).schedule_hp('lr', lr))
    ph2 = (TrainingPhase(batches*5/8).schedule_hp('lr', lr/10))
    ph3 = (TrainingPhase(batches*2/8).schedule_hp('lr', lr/100))
    model_name = f'{dataset}_{model}_{opt}-{lr}-{mom}-{wd}_bs{bs}_tr'
    learn.callback_fns += [
        partial(GeneralScheduler, phases=(ph1,ph2,ph3)),
        partial(SaveModelCallback, every='epoch', name=model_name),
    ]
    learn.fit(epochs, 1)
