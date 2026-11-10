"""Regenerate the released scheduling inputs, or verify their SHA-256 checksums."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
import formal_synthetic_dataset as v1
import formal_synthetic_dataset_v2 as v2
import academic_dataset as adapter


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')


def formal_instance(row, config):
    source = ROOT/'data/reference/brandimarte'/row['file']
    jobs = adapter.load_or_generate_benchmark(config, row['arrival_seed'], str(source))
    case = {'schema_version':'formal-synthetic-dfjsp-hagv-v2','protocol_id':'CIE_PUBLIC_MAIN_TEST_V2',
        'instance_id':row['instance_id'],'split':'formal_test','regime_id':row['base_benchmark'],
        'seeds':{'arrival_seed':row['arrival_seed']},
        'base_benchmark':{'file':row['file'],'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                          'machine_count_in_source':int(source.read_text().splitlines()[0].split()[1])},
        'environment':{**config,'n_initial':int(len(jobs)*0.7),'n_insert':len(jobs)-int(len(jobs)*0.7),
            'agv_heterogeneity_model':'nested_capacity_compatibility_with_speed',
            'transport_time_model':'transport_distance/agv_speed',
            'transport_distance_model':'abs(origin_machine-destination_machine)+1'},
        'dynamic_profile':{'arrival_process':'integer_interarrival','insertion_start':float(config['Tr']),
            'interarrival_min':20,'interarrival_max_inclusive':79,
            'due_date_rule':'arrival_time + int(1.5 * sum(mean(eligible processing times)))'},
        'jobs':[dict(job_id=int(j[0]),job_type=int(j[1]),arrival_time=float(j[2]),due_date=float(j[3]),
            operations=[{'operation_id':i,'machine_options':{str(int(m)):int(p) for m,p in op.items()}}
                        for i,op in enumerate(j[4])],priority=float(j[5]),transport_demand=float(j[6])) for j in jobs]}
    case['content_fingerprint']=v1._instance_content_fingerprint(case)
    return case


def generate(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory; released files are never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    v2.generate_dataset(output/'development')
    cfg=read(ROOT/'protocols/formal_environment.json')
    for row in read(ROOT/'protocols/formal_test_manifest.json')['instances']:
        write(output/'formal_test'/(row['instance_id']+'.json'),formal_instance(row,cfg))
    for condition,spec in read(ROOT/'protocols/generalization_conditions.json').items():
        for rep in range(1,11):
            dimensions={'id':condition,**spec['dimensions']}
            case=v2._upgrade_instance(v1._generate_instance('generalization',dimensions,rep,
                2_100_000+rep,2_200_000+rep,spec['arrival_rate'],('standard',1.30,1.70)))
            fleet=[]
            for t,count in enumerate(spec['fleet_counts']):
                for _ in range(count):
                    fleet.append({'agv_id':len(fleet),'agv_type':t,'capacity':v2.CAPACITY_BY_TYPE[t],
                                  'speed':v2.SPEED_BY_TYPE[t]})
            case['environment']['agv_fleet']=fleet
            case['environment']['n_agv']=len(fleet)
            case['condition']=condition
            case['content_fingerprint']=v1._instance_content_fingerprint(case)
            write(output/'generalization'/condition/(case['instance_id']+'.json'),case)
    count=0
    for collection in ['development/train','development/validation','formal_test','generalization']:
        for released in (ROOT/'data'/collection).rglob('*.json'):
            relative=released.relative_to(ROOT/'data')
            assert read(released)==read(output/relative), f'Regeneration mismatch: {relative}'
            count+=1
    assert count==282
    return {'regenerated_instances':count,'all_match_released_instances':True}


def verify():
    manifest=read(ROOT/'checksums.json')
    for row in manifest['files']:
        path=ROOT/row['path']
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=row['sha256']:
            raise ValueError('Checksum mismatch: '+row['path'])
    assert v2.verify_dataset(ROOT/'data/development',execute_all=False)['status']=='PASS'
    return {'verified_files':len(manifest['files']),'status':'PASS'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Empty directory for regenerated instances.')
    parser.add_argument('--verify',action='store_true',help='Verify all released file checksums.')
    args=parser.parse_args()
    if not args.verify and args.output is None:
        parser.error('Use --verify or supply --output DIRECTORY.')
    print(json.dumps(verify() if args.verify else generate(args.output),indent=2))
