"""Diagnostic: example closed-loop trajectories (expert vs pretrained vs SetONet-FT)
with achieved costs, for P2P-Cost / P2P-Dynamics / Quadrotor.

Helps sanity-check the normalized-cost metric (esp. why Quadrotor sits near 1.0).
"""
import argparse, yaml, copy, numpy as np
import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx, optax
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from src.setonet import SetONet
from src.envs.create_p2p_cost import generate_dataset
from src.envs.dynamics_models import LinearModel, LinearModelParams, PlanarQuadrotor, PlanarQuadrotorParams
from src.plotting.plot_maml_scatter import (TransferTaskDataLoader, _get_context,
    build_dynamics_normalized, NUM_TASKS, K)
from src.plotting.plot_maml_scatter_cost import (_adapt_setonet, _adapt_dynamics, _sample_dyn_context,
    FREEZE_MODE, GRAD_STEPS, NUM_DEMOS)


def rollout_traj(predict_phys, start, step_fn, stage, terminal, horizon):
    s = np.asarray(start, float); states = [s.copy()]; cost = 0.0
    for t in range(horizon):
        u = np.asarray(predict_phys(s, t), float)  # raw step index; closure handles t/H vs raw t
        cost += float(stage(s, u)); s = np.asarray(step_fn(s, u), float); states.append(s.copy())
    return np.array(states), cost + float(terminal(s))


def expert_cost(es, ea, stage, terminal):
    return sum(float(stage(es[t], ea[t])) for t in range(len(ea))) + float(terminal(es[-1]))


def do_p2p_cost(ckpt, cfgdir, seed):
    cfg = yaml.safe_load(open(Path(cfgdir) / "p2p_cost.yaml")); mc = cfg["model"]
    dt, H = 0.1, 50; Qw, Rw, Qfw = 1.0, 0.1, 10.0
    A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float); B = np.array([[0,0],[0,0],[dt,0],[0,dt]], float)
    model = SetONet(input_size_src=6, output_size_src=1, input_size_tgt=5, output_size_tgt=2,
                    **{k: mc[k] for k in ['p','phi_hidden_size','phi_output_size','rho_hidden_size','trunk_hidden_size','n_phi_layers','n_rho_layers','n_trunk_layers','aggregation_type','attention_n_heads','attention_n_tokens','use_bias']}, key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(str(Path(ckpt)/"p2p_cost"/"pretrained"/"setonet.eqx"), model)
    h = np.load(str(Path(ckpt)/"p2p_cost"/"pretrained"/"training_history.npz"))
    sm, ss = np.array(h["state_mean"]), np.array(h["state_std"]); maxa = float(h["max_action"])
    norm = {"state_mean": sm, "state_std": ss, "cost_mean": float(h["cost_mean"]), "cost_std": float(h["cost_std"])}
    np.random.seed(seed)
    ds = generate_dataset(num_goals=NUM_TASKS, trajectories_per_goal=50, horizon=H, dt=dt, Q_weight=Qw, R_weight=Rw,
        Qf_weight=Qfw, goal_range=(-5.,5.), state_range=(-10.,10.), vel_range=(-5.,5.), zero_velocity_goal=True, seed=seed+1000)
    ds["norm_stats"] = norm
    np.random.seed(seed); dl = TransferTaskDataLoader(ds, 0.25, True); dl.max_action = maxa
    tid = dl.get_task_ids()[0]; goal = np.asarray(ds["goal_states"][tid], float)
    def stage(s,u): e=s-goal; return Qw*(e@e)+Rw*(u@u)
    def term(s): e=s-goal; return Qfw*(e@e)
    step = lambda s,u: A@s + B@np.clip(u,-50,50)
    eb = dl.sample_holdout(tid, K, N=1); ts = np.asarray(eb[4]); tc = np.asarray(eb[5])*maxa
    es = ts[0,:,:-1]*ss + sm; start = es[0]; ea = tc[0]
    si, sv = _get_context(dl, tid, ds)
    ft = _adapt_setonet(model, dl, tid, ds, "ft", GRAD_STEPS, NUM_DEMOS)
    def pred(m): return lambda s,t: np.asarray(m(si, sv, jnp.array([*((s-sm)/ss), t/H])))*maxa
    return dict(goal=goal[:2], expert=np.vstack([es, es[-1]])[:, :2], idx=(0,1),
        runs={"Pre-trained": rollout_traj(pred(model), start, step, stage, term, H),
              "SetONet-FT": rollout_traj(pred(ft), start, step, stage, term, H)},
        exp_cost=expert_cost(es, ea, stage, term), title="P2P-Cost", xy=("x","y"))


def do_dyn_like(env, ckpt, cfgdir, datadir, seed, is_quad):
    cfg = yaml.safe_load(open(Path(cfgdir)/f"{env}.yaml")); mc=cfg["model"]; cw=cfg["data"]["cost_weights"]
    dt=cfg["data"]["dt"]; H=cfg["data"]["horizon"]; sd,ad=(6,2) if is_quad else (4,2)
    mk={k:mc[k] for k in ['p','phi_hidden_size','phi_output_size','rho_hidden_size','trunk_hidden_size','n_phi_layers','n_rho_layers','n_trunk_layers','aggregation_type','attention_n_heads','attention_n_tokens','use_bias']}
    base=SetONet(input_size_src=sd+ad,output_size_src=sd,input_size_tgt=sd+1,output_size_tgt=ad,**mk,key=jr.PRNGKey(0))
    model=base if is_quad else build_dynamics_normalized(base,cfg)
    model=eqx.tree_deserialise_leaves(str(Path(ckpt)/env/"pretrained"/"setonet.eqx"),model)
    am_=as_=None
    if is_quad:
        hh=np.load(str(Path(ckpt)/env/"pretrained"/"training_history.npz")); am_=np.array(hh["action_mean"]); as_=np.array(hh["action_std"])
    ds=np.load(str(Path(datadir)/env/"trajectories.npz"),allow_pickle=True); dataset={k:ds[k] for k in ds.files}
    goal=np.asarray(dataset["goal_states"][int(np.unique(dataset["goal_indices"])[0])],float)
    dynp=np.asarray(dataset["dynamics_params"],float)
    uniq=np.unique(dataset["dynamics_indices"]); np.random.seed(seed); np.random.shuffle(uniq)
    ci=list(uniq[int(len(uniq)*0.8):])[0]; ti=np.where(dataset["dynamics_indices"]==ci)[0]
    p=dynp[ci]; sim=(PlanarQuadrotor(PlanarQuadrotorParams(*p)) if is_quad else LinearModel(LinearModelParams(*p)))
    step=lambda s,u: np.asarray(sim.step(jnp.array(s),jnp.array(u),dt),float)
    if is_quad:
        gy,gz,gphi=goal[0],goal[1],goal[2]; fh=p[0]*p[3]
        def stage(s,u): return (cw["position_weight"]*((s[0]-gy)**2+(s[1]-gz)**2)+cw["velocity_weight"]*(s[3]**2+s[4]**2)+cw["angle_weight"]*((s[2]-gphi)**2)+cw["angular_velocity_weight"]*(s[5]**2)+cw["control_weight"]*((u[0]-fh)**2+u[1]**2))
        def term(s): return (cw["terminal_position_weight"]*((s[0]-gy)**2+(s[1]-gz)**2)+cw["terminal_velocity_weight"]*(s[3]**2+s[4]**2)+cw["terminal_angle_weight"]*((s[2]-gphi)**2+s[5]**2))
    else:
        gp,gv=goal[:2],goal[2:]
        def stage(s,u): return cw["position_weight"]*np.sum((s[:2]-gp)**2)+cw["velocity_weight"]*np.sum((s[2:]-gv)**2)+cw["control_weight"]*np.sum(u**2)
        def term(s): return cw["terminal_position_weight"]*np.sum((s[:2]-gp)**2)+cw["terminal_velocity_weight"]*np.sum((s[2:]-gv)**2)
    j=ti[0]; es=dataset["states"][j]; ea=dataset["actions"][j]; start=es[0]
    si,sv=_sample_dyn_context(dataset,ti,K)
    ft=_adapt_dynamics(model,dataset,ti,"trunk",GRAD_STEPS,NUM_DEMOS,am_,as_,raw_time=not is_quad)
    def pred(m):
        if is_quad: return lambda s,t: np.asarray(m(si,sv,jnp.array([*s, t/H])))*as_+am_
        return lambda s,t: np.asarray(m(si,sv,jnp.array([*s, float(t)])))
    return dict(goal=goal[:2], expert=es[:,:2], idx=(0,1),
        runs={"Pre-trained": rollout_traj(pred(model),start,step,stage,term,H),
              "SetONet-FT": rollout_traj(pred(ft),start,step,stage,term,H)},
        exp_cost=expert_cost(es,ea,stage,term), title=("Quadrotor (y,z)" if is_quad else "P2P-Dynamics"), xy=(("y","z") if is_quad else ("x","y")))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs"); ap.add_argument("--data",default="data")
    ap.add_argument("--checkpoints",default="checkpoints"); ap.add_argument("--output",default="outputs/figures/example_rollouts.png")
    ap.add_argument("--seed",type=int,default=42); a=ap.parse_args()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    panels=[do_p2p_cost(a.checkpoints,a.config,a.seed),
            do_dyn_like("p2p_dynamics",a.checkpoints,a.config,a.data,a.seed,False),
            do_dyn_like("quadrotor",a.checkpoints,a.config,a.data,a.seed,True)]
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for ax,P in zip(axes,panels):
        i,j=P["idx"]
        ax.plot(P["expert"][:,0],P["expert"][:,1],"k-",lw=2,label=f"Expert (cost {P['exp_cost']:.1f})")
        for (name,(st,c)),col in zip(P["runs"].items(),["#4477AA","#EE7733"]):
            ax.plot(st[:,i],st[:,j],"-",color=col,lw=1.5,alpha=0.9,label=f"{name} ({c/P['exp_cost']:.2f}x)")
            ax.plot(st[0,i],st[0,j],"o",color=col,ms=4)
        ax.plot(*P["goal"],"g*",ms=16,label="goal")
        ax.plot(P["expert"][0,0],P["expert"][0,1],"ks",ms=6)
        ax.set_title(P["title"]); ax.set_xlabel(P["xy"][0]); ax.set_ylabel(P["xy"][1]); ax.legend(fontsize=8)
        print(f"{P['title']:14} expert={P['exp_cost']:.2f}  " + "  ".join(f"{n}={c:.2f}({c/P['exp_cost']:.2f}x)" for n,(s,c) in P["runs"].items()))
    fig.tight_layout(); fig.savefig(a.output,dpi=130,bbox_inches="tight"); print("saved",a.output)


if __name__=="__main__": main()
