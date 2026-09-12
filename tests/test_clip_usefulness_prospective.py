import json
import ast
import hashlib
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest
import torch

from scripts.analyze_clip_usefulness_predictors import fit_rule
from src import clip_usefulness_prospective as study


def states():
    return [dict(training_seed=seed,checkpoint_step=step,full_gradient_alignment=.05 if step==100 else .01,
                 effect=-1e-5 if step==100 else 1e-5) for seed in study.SEEDS for step in study.STEPS]


def test_frozen_rules_reproduce_development_fit_and_new_seeds_are_disjoint():
    config,frozen = study.load_study()
    assert not set(study.SEEDS) & set(frozen['development_seeds'])
    for name in study.RULE_NAMES:
        assert fit_rule(frozen['states'],name) == frozen['rules'][name]
    assert config['refit_on_new_seeds'] is False
    for seed in study.SEEDS:
        _,training,protocol = study.protocols(seed,f'seed_{seed}')
        assert training['experiment']['seed'] == training['training']['seed'] == seed
        assert training['training']['epochs'] == 50 and not training['ot']['enabled']
        assert protocol['checkpoint_steps'] == study.STEPS and 'endpoint_reference_directory' not in protocol
        assert protocol['diagnostic_caption_seed'] == 42
    with pytest.raises(ValueError):
        study.protocols(42,'seed_42')


def test_predictions_do_not_depend_on_new_outcomes():
    _,frozen = study.load_study()
    original = states()
    first = study.score(original,frozen)
    changed = [{**r,'effect':-r['effect']} for r in original]
    second = study.score(changed,frozen)
    assert [p['predicted_helpful'] for p in first['predictions']] == [p['predicted_helpful'] for p in second['predictions']]
    assert all(r['correct']==5 for r in first['per_seed_scores'])
    assert first['primary_mean_balanced_accuracy_difference'] == 0
    assert len(first['predictions']) == 30


def test_single_class_and_near_zero_are_explicit():
    _,frozen = study.load_study()
    changed = [{**r,'effect':1e-8} for r in states()]
    report = study.score(changed,frozen)
    assert report['primary_mean_balanced_accuracy_difference'] is None
    assert all(r['balanced_accuracy'] is None and r['non_near_zero_count']==0 for r in report['per_seed_scores'])


@pytest.mark.parametrize('kind',['missing','duplicate','nonfinite'])
def test_bad_state_tables_fail(kind):
    _,frozen = study.load_study()
    rows = states()
    if kind=='missing':
        rows.pop()
    elif kind=='duplicate':
        rows[-1] = rows[0]
    else:
        rows[0]['effect'] = float('nan')
    with pytest.raises(AssertionError):
        study.score(rows,frozen)


def test_new_seed_observer_records_five_states_without_historical_reference(tmp_path,monkeypatch):
    config,training,protocol = study.protocols(789,'seed_789')
    holdout = set(json.loads((study.ROOT/training['dataset']['diagnostic_holdout_indices']).read_text()))
    data = SimpleNamespace(train_loader=[None]*77,train_dataset=SimpleNamespace(source_indices=[i for i in range(5994) if i not in holdout]))
    monkeypatch.setattr(study.paired,'cache_heldout',lambda *args: [])
    monkeypatch.setattr(study.paired,'encode_cached',lambda *args:(torch.ones(2,3),torch.ones(2,3),1.))
    model = torch.nn.Linear(3,3).train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    (tmp_path/'checkpoints/baseline').mkdir(parents=True)
    observer = study.ProspectiveObserver(tmp_path,config,protocol)
    observer.initialize(model=model,processor=None,data=data,device='cpu',config=training,total_steps=3850)
    observer.bind_training_state(optimizer=optimizer,scheduler=scheduler,objective=None,data=data)
    assert observer.references == {}
    for step in study.STEPS:
        observer(model=model,epoch=2,global_step=step)
        report = json.loads((tmp_path/f'results/baseline/diagnostics/{step:06d}_pulse_{step}/report.json').read_text())
        assert report['training_seed'] == 789 and 'historical_endpoint_state_exact' not in report
        assert model.training
    assert observer.seen == study.STEPS


def test_prospective_plot_and_no_drive_runner(tmp_path):
    _,frozen = study.load_study()
    rows=states()
    study.plot(tmp_path,rows,study.score(rows,frozen))
    assert (tmp_path/'prospective_usefulness.png').exists()
    source=(study.ROOT/'colabs/run_clip_usefulness_prospective.py').read_text()
    assert '25 * 1024**3' in source
    for forbidden in ['/content/drive','drive.mount','RunBackup','copy_verified']:
        assert forbidden not in source


def populate_completed(output):
    """Historical rows reused only as arithmetic/plumbing fixtures, never results."""
    config,frozen=study.load_study()
    reference=study.ROOT/'experiment_results/clip_paired_updates_2026-09'
    old_rows=[json.loads(s) for s in (reference/'paired_updates.jsonl').read_text().splitlines()]
    old_states=json.loads((reference/'checkpoint_provenance.json').read_text())
    results=output/'results'
    results.mkdir()
    write=lambda path,value: path.write_text(json.dumps(value))
    for name,value in [('protocol.json',config),('frozen_rules.json',frozen),('completion.json',dict(
        status='complete',training_seeds=study.SEEDS,checkpoint_steps=study.STEPS,total_branch_rows=720,evaluated_states=15,rules_refitted=False))]:
        write(results/name,value)
    all_states=[]
    for seed in study.SEEDS:
        source=output/f'seed_{seed}'
        directory=source/'results/paired'
        directory.mkdir(parents=True)
        rows=[{**r,'checkpoint_step':step} for step in study.STEPS for r in old_rows if r['checkpoint_step']==(1001 if step==1001 else 100)]
        (directory/'paired_updates.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        provenance=[{**old_states[1 if step==1001 else 0],'step':step} for step in study.STEPS]
        write(directory/'checkpoint_provenance.json',provenance)
        write(directory/'completion.json',dict(status='complete',rows=240,checkpoint_steps=study.STEPS))
        for state in provenance:
            stage=source/f'results/baseline/diagnostics/{state["step"]:06d}_pulse_{state["step"]}'
            stage.mkdir(parents=True)
            write(stage/'report.json',dict(training_seed=seed,features_and_scale_sha256=state['features_sha256'],
                initial_state_sha256=state['initial_state_sha256'],learning_rates=state['learning_rates']))
        all_states.extend(study.state_records(source,seed))
    write(results/'prediction_report.json',study.score(all_states,frozen))


@pytest.mark.parametrize('failure',[None,'missing_row','changed_rule','changed_prediction','changed_state'])
def test_completion_verifier_checks_all_seeds_rules_and_scores(tmp_path,failure):
    populate_completed(tmp_path)
    if failure=='missing_row':
        path=tmp_path/'seed_789/results/paired/paired_updates.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]))
    elif failure in ['changed_rule','changed_prediction']:
        path=tmp_path/'results'/('frozen_rules.json' if failure=='changed_rule' else 'prediction_report.json')
        content=json.loads(path.read_text())
        if failure=='changed_rule':
            content['rules']['checkpoint_step']['threshold']=500
        else:
            content['predictions'][0]['predicted_helpful']=not content['predictions'][0]['predicted_helpful']
        path.write_text(json.dumps(content))
    elif failure=='changed_state':
        path=tmp_path/'seed_31415/results/baseline/diagnostics/000500_pulse_500/report.json'
        content=json.loads(path.read_text())
        content['initial_state_sha256']='wrong'
        path.write_text(json.dumps(content))
    if failure:
        with pytest.raises(AssertionError):
            study.verify_completion(tmp_path)
    else:
        study.verify_completion(tmp_path)


def handoff_namespace():
    source=(study.ROOT/'colabs/clip_usefulness_prospective_one_cell.py').read_text()
    tree=ast.parse(source)
    assert isinstance(tree.body[-1],ast.Expr) and tree.body[-1].value.func.id=='run_prospective_study'
    tree.body.pop()
    namespace={'__name__':'handoff_test'}
    exec(compile(tree,'<prospective handoff>','exec'),namespace)
    return namespace


def test_one_cell_pins_immutable_runner():
    namespace=handoff_namespace()
    source=subprocess.check_output(['git','show',namespace['SOURCE_COMMIT']+':colabs/run_clip_usefulness_prospective.py'],cwd=study.ROOT)
    assert hashlib.sha256(source).hexdigest()==namespace['RUNNER_SHA256']


@pytest.mark.parametrize('changed_rules',[False,True])
def test_one_cell_redownloads_only_complete_frozen_results(tmp_path,monkeypatch,changed_rules):
    namespace=handoff_namespace()
    run='clip_usefulness_prospective_fixture'
    control=tmp_path/('otco_control_'+run)
    control.mkdir()
    archive=control/(run+'_complete.zip')
    completion=dict(status='complete',training_seeds=study.SEEDS,checkpoint_steps=study.STEPS,
                    total_branch_rows=720,evaluated_states=15,rules_refitted=changed_rules)
    with zipfile.ZipFile(archive,'w') as bundle:
        bundle.writestr(run+'/results/run_manifest.json',json.dumps(dict(source_commit=namespace['SOURCE_COMMIT'],status='complete')))
        bundle.writestr(run+'/results/completion.json',json.dumps(completion))
    downloads=[]
    monkeypatch.setitem(sys.modules,'google.colab',SimpleNamespace(files=SimpleNamespace(download=downloads.append)))
    namespace['Path']=lambda _:tmp_path
    monkeypatch.setattr(namespace['shutil'],'which',lambda _:pytest.fail('Must not start training or require GPU for re-download'))
    if changed_rules:
        with pytest.raises(RuntimeError,match='failed verification'):
            namespace['run_prospective_study']()
        assert downloads==[]
    else:
        namespace['run_prospective_study']()
        assert downloads==[str(archive)]
