# ! /usr/bin/env python3
# python3.5+
import sys
import sched
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import asyncio
# from nats.aio.client import Client as NATS # pip install asyncio-nats-client
# from nats.aio.errors import ErrConnectionClosed, ErrTimeout

import signal
import traceback
import time
import itertools as it

import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from total_utils import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from multiprocessing import Process, Manager, Array
from tensorflow.keras.utils import to_categorical

from clickhouse_driver.client import Client

def get_config_file():
    return config


def execute_ch_all(sql, param=None, with_column_types=False, **kwargs):
    cs = config['cs']
    for i in range(len(cs)):
        ch = cs[i]
        try:
            client = Client(ch['host'], port = ch['port'], send_receive_timeout=int(ch['timeout']),
                            settings={'max_threads': int(ch['thread'])})
            client.connection.force_connect()
            if client.connection.connected:
                print('[clickhouse client.execute(sql)] connected to {}'.format(ch))
                result = client.execute(sql, params=param, with_column_types=with_column_types)
                client.disconnect()
                print(ch, result)
                # return ch, result
            else:
                print('[clickhouse client.execute(sql)] cannot connected to {}'.format(ch))
                sys.exit(1)
        except Exception as err:
            logging.error(err, exc_info=1)
            sys.exit(1)


def execute_ch(sql, param=None, with_column_types=True):
    client = check_cs(0)
    if client == None:
        sys.exit(1)
    
    result = client.execute(sql, params=param, with_column_types=with_column_types)

    client.disconnect()
    return result


def check_cs(index):
    cs = config['cs']
    if index >= len(cs):
        logging.error('[clickhouse client ERROR] connect fail')
        return None
    ch = cs[index]
    try:
        client = Client(ch['host'], port=ch['port'], send_receive_timeout=int(ch['timeout']),
                        settings={'max_threads': int(ch['thread'])})
        client.connection.force_connect()
        if client.connection.connected:
            return client
        else:
            return check_cs(index + 1)
    except:
        return check_cs(index + 1)


def is_connected_all():
    """Check Clickhouse Server all Connected
    params
    ------
    None

    return
    ------
    _is_connected_all : int
        return 1 when server is all connected
    """
    cs = config['cs']
    _is_connected_all = 1
    not_connected = []
    for i in range(len(cs)):
        ch = cs[i]
        client = Client(ch['host'], port=ch['port'], send_receive_timeout=int(ch['timeout']),
                        settings={'max_threads': int(ch['thread'])})
        client.connection.force_connect()
        print('{} is connected'.format(cs[i]))
        _is_connected_all *= client.connection.connected
        client.disconnect()
        if not client.connection.connected:
            not_connected.append(cs[i])
    if _is_connected_all:
        print("clickHouse client is all connected")
    else:
        print(not_connected)
    return _is_connected_all



def execute_ch_all_return_df(sql, param=None, with_column_types=False, **kwargs):
    """The results from each server are integrated and return
    params
    ------
    sql: str
        sql
    with_column_types: Boolean
        default False
    
    return
    ------
    df: pd.Dataframe
        Target Dataframe
    """
    if is_connected_all():
        cs = config['cs']
        try:
            df = pd.DataFrame()
            for i in range(len(cs)):
                ch = cs[i]
                client = Client(ch['host'], port = ch['port'], send_receive_timeout=int(ch['timeout']),
                                settings={'max_threads': int(ch['thread'])})
                client.connection.force_connect()

                if client.connection.connected:
                    print('[clickhouse client.execute(sql)] connected to {}'.format(ch))

                    result, meta = client.execute(sql, params=param, with_column_types=True)
                    client.disconnect()
                    feats = [m[0] for m in meta]
                    _df = pd.DataFrame(result, columns=feats)
                    _df['clickhouse_server'] = ch['host']
                    df = df.append(_df)
                else:
                    print("some clickhouse client is not connected")
                    sys.exit(1)
            return df.reset_index(drop=True)
        except Exception as err:
            logging.error(err, exc_info=1)
            sys.exit(1)



def getCols(_table):
    (dp.table) = _table.split('.')
    return np.array(execute_ch("""select name from system.columns where database = '{}' and table = '{}' and default_kind != 'MATERIALIZED'""".format(db,table)))[:,0]

# def exceute_ch(sql, param=None):
#     client = check_cs(0)
#     if client == None:
#         sys.exit(1)
    
#     result = client.execute(sql, param)

#     client.disconnect()
#     return result

# def check_cs(index):
#     cs = config['cs']
#     if index >= len(cs):
#         logger.error('[clickhouse client ERROR] connect fail')
#         return None
#     ch = cs[index]
#     try:
#         client = Client(ch["host"], port=ch["port"], send_receive_timeout=int(ch['timeout']), settings={"max_threads":int(ch['thread'])})
#         client.connection.force_connect()
#         if client.connection.connected:
#             logger.info('[clickhouse client.execute(sql)] connected to {}'.format(ch))
#             return client
#         else:
#             return check_cs(index+1)
#     except:
#         return check_cs(index+1)

async def execute_async_ch(sql, param=None):
    client = check_async_cs(0)
    if client == None:
        sys.exit(1)
    
    result = await client.execute(sql, param)

    client.disconnect()
    return result

def check_async_cs(index):
    cs = config['cs']
    if index >= len(cs):
        logger.error('[clickhouse async client ERROR] connect fail')
        return None
    ch = cs[index]

    try:
        client = AsyncClient(ch["host"], port=ch["port"], send_receive_timeout=int(ch['timeout']), settings={"max_threads":int(ch['thread'])})
        client.connection.force_connect()
        if client.connection.connected:
            logger.info('[clickhouse async client.execute(sql)] connected to {}'.format(ch))
            return client
        else:
            return check_async_cs(index+1)
    except:
        return check_async_cs(index+1)

async def _run_in_executor(executor, func, *args, **kwargs):
    if kwargs:
        func = partial(func, **kwargs)
    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(executor, func, *args)

class AsyncClient(Client):
    def __init__(self, *args, **kwargs):
        self.executor = ThreadPoolExecutor(max_workers = 1)
        super(AsyncClient, self).__init__(*args, **kwargs)

    async def execute(self, *args, **kwargs):
        return await _run_in_executor(self.executor, super(AsyncClient, self).execute, *args, **kwargs)

def get_specs(spec_dict, ips, i, query_bs, param):
    try:
        clus_data = []
        j = 0
        for j in range(int(len(ips)/query_bs)):
            try:
                clus_res = execute_ch(dns_cluster_sql(param, ips[j*query_bs:(j+1)*query_bs]))
                clus_data = clus_data + clus_res
            except Exception as err:
                traceback.print_exc()
        if int(len(ips)/query_bs) > j*query_bs or len(ips) < query_bs:
            try:
                clus_res = exceute_ch(dns_cluster_sql(param, ips[j*query_bs:]))
                clus_data = clus_data + clus_res
            except Exception as err:
                traceback.print_exc()
        final_result = np.mean(np.array(clus_data), 0)
        # final_result[-1] = final_result[-1]/len(ips)
        # final_result[-1] = final_result[-2]/len(ips)
        # final_result[-1] = final_result[-3]/len(ips)
        spec_dict[i] = final_result
    except Exception as err:
        traceback.print_exc()

def EF_AST(ip_tuple, query_bs, X_data, num_meta, sstart, param, train=True):
    logger.info(len(ip_tuple))
    logger.info(int(len(ip_tuple)/query_bs))
    ast_data = []
    j = 0
    for j in range(int(len(ip_tuple)/query_bs)):
        try:
            ast_r = ast(ip_tuple[j*query_bs:(j+1)*query_bs])
            ast_data = ast_data + ast_r
        except:
            logger.info('AST---------------------------'+str(j))
            logger.info(ip_tuple[j*query_bs:(j+1)*query_bs])
    if int(len(ip_tuple)/query_bs) > (j+1)*query_bs:
        try:
            ast_r = ast(ip_tuple[(j+1)*query_bs:])
            ast_data = ast_data + ast_r
        except:
            pass
    elif init(len(ip_tuple)/query_bs) == 0:
        try:
            ast_r = ast(ip_tuple)
            ast_data = ast_data + ast_r
        except:
            pass
    
    j = 0
    efl_data = []
    for j in range(int(len(ip_tuple)/query_bs)):
        try:
            efl_r = efl(param, ip_tuple[j*query_bs:(j+1)*query_bs])
            efl_data = efl_data + efl_r
        except Exception as err:
            logger.error(err)
            logger.info('EFLOG----------------------------'+str(j))
            logger.info(ip_tuple[j*query_bs:(j+1)*query_bs])
    if int(len(ip_tuple)/query_bs) > (j+1)*query_bs:
        try:
            efl_r = efl(param, ip_tuple[(j+1)*query_bs:])
            efl_data = efl_data + efl_r
        except:
            pass
    elif int(len(ip_tuple)/query_bs) == 0:
        try:
            efl_r = efl(param, ip_tuple)
            efl_data = efl_data + efl_r
        except:
            pass

            # ast_data = np.array(ast(ip_tuple))
    logger.info('[LENGTH AST] : {}'.format(len(ast_data)))
    logger.info('[LENGTH EFL] : {}'.format(len(efl_data)))

    data_df = pd.DataFrame(X_data, columns=[str(i) for i in range(X_data.shape[-1])])
    if len(ast_data) == 0:
        ast_df = pd.DataFrame([[X_data[0][2], 0] for n in range(1)], columns=[str(i+X_data.shape[-1]) for i in range(2)])
    else:
        ast_df = pd.DataFrame(ast_data, columns=[str(i+X_data.shape[-1]) for i in range(len(ast_data[0]))])
    if len(efl_data) == 0:
        efl_df = pd.DataFrame([[X_data[n][2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for n in range(len(X_data))], columns=[str(i+X_data.shape[-1]+2) for i in range(11)])
    else:
        efl_df = pd.DataFrame(efl_data, columns=[str(i+X_data.shape[-1]+2) for i in range(len(efl_data[0]))])
    
    logger.info(len(ast_data))
    logger.info(len(ast_df))
    logger.info(len(efl_data))
    logger.info(len(efl_df))
    logger.info(len(data_df))

    logger.info(data_df.info())
    logger.info(ast_df.info())
    logger.info(efl_df.info())
    # joined = data_df.merge(ast_df, how='left', left_on='2', right_on=str(X_data.shape[-1]))
    # logger.info(joined.info())
    # joined = joined.merge(efl_df, how='left', left_on='2', right_on=str(X_data.shape[-1]+2))
    logger.info(data_df['2'][:10])
    logger.info(ast_df[str(X_data.shape[-1])][:10])

    joined = pd.merge(data_df, ast_df, how='left', left_on='2', right_on=str(X_data.shape[-1]))
    logger.info(joined.info())

    joined = pd.merge(joined, efl_df, how='left', left_on='2', right_on=str(X_data.shape[-1]+2))

    logger.info(joined.info())
    cols = [str(i) for i in range(num_meta+sstart)]+[str(X_data.shape[-1]+i) for i in range(2)]+[str(X_data.shape[-1]+i+2) for i in range(11)]+[str(num_meta+sstart+i) for i in range(X_data.shape[-1]-(num_meta+sstart))]

    joined = joined[cols]
    joined = joined.drop([str(X_data.shape[-1]), str(X_data.shape[-1]+2)], axis=1).fillna(0)
    if train:
        joined[(joined['42'] > 0) & (joined['40'] == 0) & (joined['38'] == 0) & (joined['36'] == 0) & (joined['34'] == 0)]['30'] = 0
    
    return joined.values, 2, 11


def transform_with_fb(total_list, data, fb_array, http_last_dim, b, http_doc_dict, vector_size, num_meta, sstart, sfinish):
    try:
        array = np.zero((http_last_dim))
        array[:sstart] = data[num_meta:num_meta+sstart]
        for st in range(sfinish-sstart):
            try:
                trans_array = np.array(http_doc_dict[sstart+st].transform([' '.join(str(m).lower() for m in data[sstart+st+num_meta])]).todense()).reshape(-1,)
                array[sstart+(vector_size*st):sstart+(vector_size*(st+1))] = trans_array
            except Exception as err:
                traceback.print_exc()
        if similarity(array, fb_array):
            total_list.append(np.concatenate((data[:num_meta], array), 0))
        if (b+1)%10000 == 0:
            logger.info('[TRANSFORM PROCESS] : {}'.format(b+1))
    except Exception as err:
        traceback.print_exc()

def transform(total_list, data, http_last_dim, b, http_doc_dict, vector_size, num_meta, sstart, sfinish):
    try:
        array = np.zeros((http_last_dim))
        array[:sstart] = data[num_meta:num_meta+sstart]
        for st in range(sfinish-sstart):
            try:
                array[sstart+(vector_size*st):sstart+(vector_size*(st+1))] = np.array(http_doc_dict[sstart+st].transform([' '.join(str(m).lower() for m in data[sstart+st+num_meta])]).todense()).reshape(-1,)
            except Exception as err:
                traceback.print_exc()
        if (b+1)%10000 == 0:
            logger.info('[TRANSFORM PROCESS] : {}'.format(b+1))
        total_list.append(np.concatenate((data[:num_meta], array), 0))
    except Exception as err:
        traceback.print_exc()

def transform_super(total_list, data, http_last_dim, b, http_doc_dict, vector_size, num_meta, sstart, sfinish):
    try:
        array = np.zeros((http_last_dim))
        array[:sstart] = data[num_meta:num_meta+sstart]
        for st in range(sfinish-sstart):
            try:
                transformed = np.array(http_doc_dict[sstart+st].transform([' '.join(str(m).lower() for m in data[sstart+st+num_meta])]).todense()).reshape(-1,)
                padding = np.zeros(vector_size-len(transformed))
                array[sstart+(vector_size*st):sstart+(vector_size*(st+1))] = np.concatenate((transformed, padding), 0)
                # array[sstart+(vector_size*st):sstart+(vector_size*(st+1))] = np.array(http_doc_dict[sstart+st].transform([' 'join(str(m).lower() for m in data[sstart+st+num_meta])]).todense()).reshape(-1,)
            except Exception as err:
                logger.info(traceback.print_exc())
        if (b+1)%10000 == 0:
            logger.info('[TRANSFORM PROCESS] : {}'.format(b+1))
        total_list.append(np.concatenate((data[:num_meta], array, data[-1:]), 0))
    except Exception as err:
        traceback.print_exc()

def execute_client(select_result, sql):
    try:
        res, _ = execute_ch(sql)
        logger.info(len(res))
        logger.info(res[0][-1])
        select_result.extend(res)
    except Exception as err:
        traceback.print_exc()

def optimize_model(model, eps, bs, data):
    try:
        model.optimize(epoches=eps, bs=bs, data=data)
        logger.info('OPTIMIZATION DONE')
    except Exception as err:
        traceback.print_exc()

def optimize_ae(model, eps, bs, xdata, ydata):
    try:
        model.optimize(epoches=eps, bs=bs, X=xdata, Y=ydata)
        model.save()
        logger.info('AE OPTIMIZATION DONE')
    except Exception as err:
        traceback.print_exc()

def feedback(m_id):
    try:
        res = execute_ch(feedback_sql(m_id))
        result = []
        if m_id == 40 or m_id ==41 or m_id == 42:
            for i in res:
                r = list(execute_ch(dns_feedback(i[0], i[1], i[2]))[0]) + [i[4]]
                result.append(r)
        elif m_id == 30:
            for i in res:
                r = list(execute_ch(http_feedback(i[0], i[1], i[2]))[0]) + [i[4]]
                result.append(r)
        elif m_id == 1:
            for i in res:
                r = list(execute_ch(nbad_feedback(i[0], i[1], i[2]))[0]) + [i[4]]
                result.append(r)
        return result
    except Exception as err:
        traceback.print_exc()
        return []

def ast(ips):
    try:
        res = execute_ch(ast_ip(ips))
        return res
    except Exception as err:
        traceback.print_exc()

def efl(param, ips):
    try:
        res = execute_ch(eflog_login(param, ips))
        return res
    except Exception as err:
        traceback.print_exc()

class BaseComponent(object):
    # def __init__(self, nc, loop, model_id, model_name):
    def __init__(self, loop, model_id, model_name):
        # self.nc = nc
        self.loop = loop
        
        self.model_id = model_id
        self.model_name = model_name
        self.update_topic = 'update_{}'.format(model_id)

        self.model = None
        self.http_doc_dict = None

#         if not os.path.exists(pwd+'/logs/'):
#             logger.info('create directory: {}'.format(pwd+'/logs/'))
#             os.makedirs(pwd+'/logs/')
#         if not os.path.exists(pwd+'/graph/'):
#             logger.info('create directory: {}'.format(pwd+'/graph/'))
#             os.makedirs(pwd+'/graph/')
#         if not os.path.exists(pwd+'/check/'):
#             logger.info('create directory: {}'.format(pwd+'/check/'))
#             os.makedirs(pwd+'/check/')
#         if not os.path.exists(pwd+'/obj/'):
#             logger.info('create directory: {}'.format(pwd+'/obj/'))
#             os.makedirs(pwd+'/obj/')

        # for sig in ('SIGINT', 'SIGTERM'):
        #     self.loop.add_signal_handler(getattr(signal, sig), self.signal_handler)

    def init(self):
        pass

    # async def closed_cb(self):
    #     logging.warning('Connection to NATS is closed.')
    #     await asyncio.sleep(0.1, loop=self.loop)
    #     self.loop.stop()
    #     await self.nc.close()

    # def signal_handler(self):
    #     if self.nc.is_closed:
    #         return
    #     logging.warning('Disconnecting...')
    #     self.loop.create_task(self.nc.close())

    async def run(self, param):
        pass

    async def update(self, msg):
        model_id = msg.data.decode()
        self.model.load()
        logger.info('[UPDATE] model update: {}'.format(model_id))

    async def run_scheduler(self, mode):
        try:
            self.logger = set_logger(mode, self.model_id)
            now = datetime.datetime.now() - self.model_config['now_delta'] + datetime.timedelta(hours=9)
            prev = now - self.model_config['prev_delta']
            prev_month = now - datetime.timedelta(weeks=8)
            prev_day = now - datetime.timedelta(days=1)
            logger.info('[{}({})] {} ~ {}'.format(mode, self.model_name, prev, now))
            param = {
                'logdate_s': prev.strftime('%Y-%m-%d'),
                'logdate_e': now.strftime('%Y-%m-%d'),
                'logtime_s': prev.strftime('%Y-%m-%d %H:%M:%S'),
                'logtime_e': now.strftime('%Y-%m-%d %H:%M:%S'),
                'logdate_m': prev_month.strftime('%Y-%m-%d'),
                'logdate_day': prev_day.strftime('%Y-%m-%d'),
#                 'result_table': self.model_config['result_table'],
#                 'cluster_table': self.model_config['cluster_table'],
#                 'model_hist': self.model_config['history_table'],
                'toStartOf': getToStartOf(self.model_config['crontab'])
            }

            logger.info('[{}({})] param: {}'.format(mode, self.model_name, param))
            task = [self.run(param)]
            for f in asyncio.as_completed(task):
                task_result = await f
                logger.info('[{}({})] result: {}'.format(mode, self.model_name, task_result))
                if task_result == None:
                    sys.exit(1)
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def start_train(self):
        try:
            # options = {
            #     'servers': ["nats://{}:{}".format(str(config['nats']['host']), str(config['nats']['port']))],
            #     'io_loop':self.loop,
            #     'closed_cb':self.closed_cb
            #     ############# max_payload ###########
            # }
            # await self.nc.connect(**options)
            # logger.info('Connected to NATS at {}...'.format(self.nc.connected_url.netloc))

            self.model_config = getConfig('TRAIN', self.model_id, self.model_name)

            scheduler = AsyncIOScheduler()
            scheduler.add_job(self.run_scheduler, CronTrigger.from_crontab(self.model_config['crontab']), ['TRAIN'], misfire_grace_time=3600, max_instances=20)
            scheduler.start()
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def test_train(self):
        try:
            # options = {
            #     'servers': ["nats://{}:{}".format(str(config['nats']['host']), str(config['nats']['port']))],
            #     'io_loop':self.loop,
            #     'closed_cb':self.closed_cb
            #     ############# max_payload ###########
            # }
            # await self.nc.connect(**options)
            # logger.info('Connected to NATS at {}...'.format(self.nc.connected_url.netloc))

            self.model_config = getConfig('TEST_TRAIN', self.model_id, self.model_name)
            await self.run_scheduler('TEST_TRAIN')
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def start_pred(self):
        try:
            # options = {
            #     'servers': ["nats://{}:{}".format(str(config['nats']['host']), str(config['nats']['port']))],
            #     'io_loop':self.loop,
            #     'closed_cb':self.closed_cb
            #     ############# max_payload ###########
            # }
            # await self.nc.connect(**options)
            # logger.info('Connected to NATS at {}...'.format(self.nc.connected_url.netloc))

            # await self.nc.subscribe(self.update_topic, cb=self.update)

            self.model_config = getConfig('PREDICTION', self.model_id, self.model_name, False)

            scheduler = AsyncIOScheduler()
            scheduler.add_job(self.run_scheduler, CronTrigger.from_crontab(self.model_config['crontab']), ['PREDICTION'], misfire_grace_time=3600, max_instances=20)
            scheduler.start()
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())

    async def test_pred(self):
        try:
            # options = {
            #     'servers': ["nats://{}:{}".format(str(config['nats']['host']), str(config['nats']['port']))],
            #     'io_loop':self.loop,
            #     'closed_cb':self.closed_cb
            #     ############# max_payload ###########
            # }
            # await self.nc.connect(**options)
            # logger.info('Connected to NATS at {}...'.format(self.nc.connected_url.netloc))

            self.model_config = getConfig('TEST_PREDICTION', self.model_id, self.model_name, False)

            await self.run_scheduler('TEST_PREDICTION')
        except Exception as err:
            logger.error(err)
            logger.error(traceback.print_exc())