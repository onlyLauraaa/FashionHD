import os
import argparse
from time import localtime, strftime, time
from sklearn.model_selection import train_test_split

from utils.utils import *
from model.layers import *
from model.graphsage import *
from RL.rl_model import *
from model.model import OneLayerRio
from model.model import TwoLayerRio


parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='amazon', help='The dataset name.')
parser.add_argument('--log_path', default='log/', type=str, help="Path of results")

parser.add_argument('--inter', type=str, default='noiseaware',


# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb_size', type=int, default=64, help='Node embedding size at the last layer.')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
parser.add_argument('--test_epochs', type=int, default=3, help='Epoch interval to run test set.')
parser.add_argument('--test_ratio', type=float, default=0.6, help='Test set size.')
parser.add_argument('--under_sample', type=int, default=1, help='Under-sampling scale.')

parser.add_argument('--no_relproj', action='store_true', 
                    help='Ablate relation-specific projection layers')
parser.add_argument('--lambda_noise', type=float, default=0.01, 
                    help='Weight for noise regularization (L1 on relation gates)')
if __name__ == '__main__':

  

  
    args = parser.parse_args()


   
    log_save_path = args.log_path + 'log_' + strftime("%m%d%H%M%S", localtime())
    os.mkdir(log_save_path)
    print("Log save path:  ", log_save_path, flush=True)



    args.cuda = args.use_cuda and torch.cuda.is_available()
    print("CUDA:  " + str(args.cuda), flush=True)

  
    homo, relations, feat_data, labels, index = load_data(args.data)
    print("Running on:  " + str(args.data), flush=True)
    print("The number of relations:  " + str(len(relations)), flush=True)

  
    np.random.seed(args.seed)
    random.seed(args.seed)
    idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
                                                            test_size=0.6, random_state=2, shuffle=True)

    
    train_pos, train_neg = pos_neg_split(idx_train, y_train)

    
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    feat_data = normalize(feat_data)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if args.cuda:
        features.cuda()

    
    width_rl = [args.ALPHA for r in range(len(relations))]
    height_rl = [math.ceil(pow(len(max(relations[r].values(), key=len)), 1 / width_rl[r]))
                 for r in range(len(relations))]
    print('Width of each relation tree:  ' + str(width_rl), flush=True)
    print('Height of each relation tree:  ' + str(height_rl), flush=True)

   
    print('Model:  {0}, Inter-AGG:  {1}, emb_size:  {2}.'.format(args.model, args.inter, args.emb_size))
    if args.model == 'RIO':
        adj_lists = relations
        intra_aggs = [IntraAgg(features, feat_data.shape[1], cuda=args.cuda) for r in range(len(relations))]
        
        from model.model import RelationGate  


        relation_gate = None
        if args.inter == 'noiseaware':
            relation_gate = RelationGate(
                embed_dim=args.emb_size,
                num_relations=len(intra_aggs)
            ).to(args.device)

       
        inter1 = InterAgg(
            width_rl, height_rl, args.device, args.LR, args.GAMMA, args.stop_num,
            features, feat_data.shape[1], args.emb_size, adj_lists,
            intra_aggs, inter=args.inter,
            relation_gate=relation_gate,       
            cuda=args.cuda,
            use_relation_proj=not args.no_relproj
        )

        inter2 = InterAgg(
            width_rl, height_rl, args.device, args.LR, args.GAMMA, args.stop_num,
            features, feat_data.shape[1], args.emb_size, adj_lists,
            intra_aggs, inter=args.inter,
            relation_gate=relation_gate,        
            cuda=args.cuda,
            use_relation_proj=not args.no_relproj
        )

       
        last_label_scores = torch.zeros(feat_data.shape[0], 2).to(args.device)  

        gnn_model = TwoLayerRio(
            num_classes=2,
            inter1=inter1,
            inter2=inter2,
            lambda_1=args.lambda_1,
            last_label_scores=last_label_scores,
            cl_weight=getattr(args, 'cl_weight', 0.1),
            lambda_noise=getattr(args, 'lambda_noise', 0.01)
        ).to(args.device)

      
        gnn_model.no_resatt = getattr(args, 'no_resatt', False)
    elif args.model == 'SAGE':
        if homo is None:    
            raise ValueError("GraphSAGE requires a homogeneous graph ('homo'), but none was provided.")
    
        adj_lists = homo
        agg1 = MeanAggregator(features, cuda=args.cuda)
        enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)
        # the vanilla GraphSAGE model as baseline
        enc1.num_samples = 5
        gnn_model = GraphSage(2, enc1)

    if args.cuda:
        gnn_model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
                                 weight_decay=args.lambda_2)

    gnn_auc_train = 0
    start_all_time = time()

    print("\nTrainable parameters:")
    for name, param in gnn_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {tuple(param.shape)}")




   
    for epoch in range(args.num_epochs):
        print('\n+------------------------------------------------------------------------------------------+\n'
              '                                        Epoch {0}                                               '
              '\n+------------------------------------------------------------------------------------------+\n'.
              format(epoch), flush=True
              )
       
        sampled_idx_train = undersample(train_pos, train_neg, scale=args.under_sample)
        rd.shuffle(sampled_idx_train)

     
        num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
        if args.model == 'RIO':
            inter1.batch_num = num_batches
            inter1.auc = gnn_auc_train

        loss = 0.0
        epoch_time = 0

      
        for batch in range(num_batches):
            start_time = time()
            i_start = batch * args.batch_size
            i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
            batch_nodes = sampled_idx_train[i_start:i_end]
            batch_label = labels[np.array(batch_nodes)]
            optimizer.zero_grad()
            if args.cuda:
                loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
            else:
                loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
            loss.backward()
            optimizer.step()
            end_time = time()
            epoch_time += end_time - start_time
            loss += loss.item()
        
     
                

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', flush=True)
        print('Loss: {0}, time: {1}s'.format(loss.item() / num_batches, epoch_time), flush=True)



        relation_weights = None
        rel_mean = None
        rel_std  = None
        rel_entropy = None

        if hasattr(inter1, "relation_gate") and inter1.relation_gate is not None:
            with torch.no_grad():
                W_r = inter1.relation_gate.W_r.detach().cpu().numpy()  # shape: (R, D)
                
    
                if W_r.ndim == 2:
                    relation_weights = W_r.tolist()
                    rel_mean = np.mean(W_r, axis=1)   
                    rel_std  = np.std(W_r, axis=1)   
                    probs = rel_mean / (rel_mean.sum() + 1e-8)
                    rel_entropy = -np.sum(probs * np.log(probs + 1e-8))
                else: 
                    rel_mean = np.array([W_r.mean()] * inter1.relation_gate.W_r.shape[0])
                    rel_std  = np.array([W_r.std()] * inter1.relation_gate.W_r.shape[0])
                    rel_entropy = 0.0
        else:
            num_relations = len(inter1.intra_aggs)
            rel_mean = np.zeros(num_relations)
            rel_std  = np.zeros(num_relations)
            rel_entropy = 0.0

        with open(relation_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for rid, (m, s) in enumerate(zip(rel_mean, rel_std)):
                writer.writerow([epoch, rid, m, s])

    # end
    print('\n+------------------------------------------------------------------------------------------+\n')
    end_all_time = time()
    total_epoch_time = end_all_time - start_all_time
    print('Total time spent:  ' + str(total_epoch_time), flush=True)
    print('Total epoch:  ' + str(epoch), flush=True)