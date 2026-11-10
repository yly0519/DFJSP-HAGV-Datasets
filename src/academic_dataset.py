import numpy as np
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _job_record(cfg, job_id, job_type, arrival, due_date, operations, priority):
    """Build a legacy or capacity-aware job tuple from the active protocol."""
    record = (job_id, job_type, arrival, due_date, operations, priority)
    if bool(cfg.get('agv_capacity_speed_heterogeneity_enabled', False)):
        return record + (float(job_type) + 1.0,)
    return record


def resolve_benchmark_path(file_path):
    """Resolve bundled benchmark names independently of the process cwd."""
    if not file_path:
        return file_path
    expanded = os.path.expanduser(os.fspath(file_path))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(PROJECT_ROOT, expanded))


def generate_poisson_training_jobs(cfg, arrival_rate=None, seed=None):
    """
    【学术权威训练集】: 采用泊松过程模拟插单。
    """
    if seed is not None:
        np.random.seed(seed)

    rate = arrival_rate if arrival_rate else cfg.get('arrival_rate', 0.15)
    jobs_data = []
    current_time = 0.0
    total_jobs = cfg['n_initial'] + cfg['n_insert']

    for i in range(total_jobs):
        if i < cfg['n_initial']:
            arr_time = 0
        else:
            if current_time < cfg['Tr']:
                current_time = cfg['Tr']
            inter_arrival = np.random.exponential(1.0 / rate)
            current_time += inter_arrival
            arr_time = int(current_time)

        j_type = np.random.randint(0, cfg['agv_types'])
        num_ops = np.random.randint(max(2, cfg['m'] // 2), cfg['m'] + 1)

        total_processing_time = 0
        ops_data = []
        for op in range(num_ops):
            num_options = np.random.randint(1, cfg['m'] + 1)
            chosen_machines = np.random.choice(range(cfg['m']), num_options, replace=False)
            m_opts = {mac: np.random.randint(10, 101) for mac in chosen_machines}
            ops_data.append(m_opts)
            total_processing_time += np.mean(list(m_opts.values()))

        urgency_factor = np.random.uniform(1.2, 2.0)
        due_date = arr_time + int(total_processing_time * urgency_factor)
        priority = np.round(np.random.uniform(0.5, 3.0), 1)
        jobs_data.append(_job_record(
            cfg, i, j_type, arr_time, due_date, ops_data, priority
        ))

    return jobs_data


def load_or_generate_benchmark(cfg, seed, file_path="MK01.txt"):
    np.random.seed(seed)
    jobs_data = []
    file_path = resolve_benchmark_path(file_path)

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('%')]
        header = lines[0].split()
        num_jobs = int(header[0])
        insert_start_idx = int(num_jobs * 0.7)

        machine_ids = []
        for raw_line in lines[1:num_jobs + 1]:
            raw = list(map(int, raw_line.split()))
            idx = 1
            for _ in range(raw[0]):
                num_options = raw[idx]
                idx += 1
                for _ in range(num_options):
                    machine_ids.append(raw[idx])
                    idx += 2
        zero_based_machine_ids = 0 in machine_ids

        current_arr_time = cfg['Tr']
        for i in range(num_jobs):
            line_data = list(map(int, lines[i + 1].split()))
            num_ops = line_data[0]
            ops_data = []
            idx = 1
            total_p_time = 0
            for op in range(num_ops):
                num_options = line_data[idx]
                idx += 1
                m_opts = {}
                for _ in range(num_options):
                    raw_mac_id = line_data[idx]
                    mac_id = raw_mac_id if zero_based_machine_ids else raw_mac_id - 1
                    p_time = line_data[idx + 1]
                    m_opts[mac_id] = p_time
                    idx += 2
                ops_data.append(m_opts)
                total_p_time += np.mean(list(m_opts.values()))

            j_type = np.random.randint(0, cfg['agv_types'])
            if i >= insert_start_idx:
                current_arr_time += np.random.randint(20, 80)
                arr_time = current_arr_time
            else:
                arr_time = 0

            due_date = arr_time + int(total_p_time * 1.5)
            jobs_data.append(_job_record(
                cfg, i, j_type, arr_time, due_date, ops_data, 1.0
            ))
    else:
        total_jobs = cfg['n_initial'] + cfg['n_insert']
        current_arr_time = cfg['Tr']
        for i in range(total_jobs):
            j_type = np.random.randint(0, cfg['agv_types'])
            num_ops = np.random.randint(cfg['m'] - 2, cfg['m'] + 1)
            total_processing_time = 0
            ops_data = []
            for op in range(num_ops):
                num_options = np.random.randint(2, cfg['m'] // 2 + 2)
                chosen_machines = np.random.choice(range(cfg['m']), num_options, replace=False)
                m_opts = {mac: np.random.randint(15, 80) for mac in chosen_machines}
                ops_data.append(m_opts)
                total_processing_time += np.mean(list(m_opts.values()))

            if i < cfg['n_initial']:
                arr_time = 0
            else:
                current_arr_time += np.random.randint(10, 50)
                arr_time = current_arr_time

            due_date = arr_time + int(total_processing_time * 1.3)
            jobs_data.append(_job_record(
                cfg, i, j_type, arr_time, due_date, ops_data, 1.0
            ))

    return jobs_data
