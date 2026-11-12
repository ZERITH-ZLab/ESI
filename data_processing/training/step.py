"""Native flow-matching objective, with temporal loss masking only."""
import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax


def masked_mean(step_loss, mask):
    """Equal weight per window, irrespective of its supervised prefix length.

    Empty masks produce NaN to fail the training finite check, never zero loss.
    Gradients of this reducer w.r.t. masked step losses are exactly zero.
    This does NOT remove their noisy action tokens from model attention.
    """
    if step_loss.ndim != 2 or mask.shape != step_loss.shape or mask.dtype != jnp.bool_:
        raise ValueError('Expected step_loss [B,H] and boolean mask [B,H]')
    counts=jnp.sum(mask,axis=-1)
    losses=jnp.sum(jnp.where(mask,step_loss,0.),axis=-1)/jnp.maximum(counts,1)
    return jnp.mean(jnp.where(counts>0,losses,jnp.nan))


def train_step(config, rng, state, batches):
    """batches has leading [accumulation, micro_batch, ...] axes.

    Average microbatch gradients FIRST, then clip/Adam once. Model compute_loss,
    real masked tails, action-dimension reduction and attention are unchanged.
    """
    if state.ema_decay is not None:
        raise ValueError('This training entry explicitly disables EMA')
    params=state.params.filter(config.trainable_filter)
    count=batches.actions.shape[0]
    base_rng=jax.random.fold_in(rng,state.step)
    def micro(carry,item):
        total,loss_sum=carry
        i,batch=item
        model=nnx.merge(state.model_def,state.params)
        model.train()
        def loss_fn(model):
            observation,actions,mask=batch
            losses=model.compute_loss(jax.random.fold_in(base_rng,i),observation,actions,train=True)
            return masked_mean(losses,mask)
        loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,config.trainable_filter))(model)
        return (jax.tree.map(lambda a,b:a+b,total,grads),loss_sum+loss),None
    zeros=jax.tree.map(jnp.zeros_like,params)
    (total,loss),_=jax.lax.scan(micro,(zeros,jnp.asarray(0.,jnp.float32)),(jnp.arange(count),batches))
    grads=jax.tree.map(lambda x:x/count,total)
    updates,opt_state=state.tx.update(grads,state.opt_state,params)
    model=nnx.merge(state.model_def,state.params)
    nnx.update(model,optax.apply_updates(params,updates))
    next_state=dataclasses.replace(state,step=state.step+1,params=nnx.state(model),opt_state=opt_state)
    return next_state,dict(loss=loss/count,grad_norm=optax.global_norm(grads),
        update_norm=optax.global_norm(updates),valid_steps=jnp.mean(jnp.sum(batches.action_loss_mask,axis=-1)))
