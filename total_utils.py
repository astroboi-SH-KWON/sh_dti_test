import os
import json
import pickle
import random
import shutil
import pymysql
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import logging.config
from multiprocessing import Process, Manager
from logging.handlers import QueueHandler, QueueListener
import queue

pwd = os.getcwd()
# pwd = '/home/ctilab/workspace/lumi/sh/test'
print(pwd)

#logging
if not os.path.exists(pwd+'/logs'):
    os.makedirs(pwd+'/logs')

#config
with open(pwd+'/conf/config.json') as f:
    config = json.loads(json.dumps(json.load(f)))

def random_combination(corpus, pool, len1, length):
    n = len(pool)
    for i in range(length):
        indices = random.sample(range(n), random.randint(len1, int(n/2)))
        new_sent = ' '.join(pool[i] for i in indices)
        if type(corpus)==set:
            corpus.add(new_sent)
        elif type(corpus)==list:
            corpus.append(new_sent)

def set_logger(mode, model_id, defult=False):
    if defult:
        logconf = json.loads(json.dumps(config['logging']).replace('{DIR}', pwd).replace('{MODE}', 'default').replace('{MODEL_ID}', 'ai'))
    else:
        logconf = json.loads(json.dumps(config['logging']).replace('{DIR}', pwd).replace('{MODE}', mode).replace('{MODEL_ID}', str(model_id)))
    logging.config.dictConfig(logconf)
    logger = logging.getLogger()
    logger.propagate = False
    #log queue
    q = queue.Queue(-1) #unlimit
    q_handler = QueueHandler(q)
    q_listener = QueueListener(q, logger.handlers)
    q_listener.start()
    logger.info('logging config : {}'.format(logconf))

    return logger

logger = set_logger(None, None, True)

def isExistModel(model_id):
    conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
    curs = conn.cursor()
    sql = 'select model_name from model_meta where model_id = {}'.format(model_id)
    result = curs.execute(sql)

    if result == 0:
        model_name = None
    else:
        model_name = list(curs.fetchone())[0]

    conn.close()
    return model_name

def getConfig(mode, model_id, model_name, train=True):
    conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
    curs = conn.cursor()
    sql = 'select config from model_meta where model_id = {}'.format(model_id)
    curs.execute(sql)
    result = list(curs.fetchone())[0]
    print(result)
    if train:
        model_config = json.loads(result)['train']
    else:
        model_config = json.loads(result)['predict']
    conn.close()

    model_config['now_delta'] = getDelta(model_config['now_delta'])
    model_config['prev_delta'] = getDelta(model_config['prev_delta'])
    logger.info('[{}({})] model config : {}'.format(mode, model_name, model_config))
    return model_config

def getToStartOf(crontab):
    try:
        m, h, d, M, y = crontab.split(' ')
        if m == '*/1' or crontab == '* * * * *':
            return 'toStartOfMinute'
        elif m == '*/5':
            return 'toStartOfFiveMinute'
        elif m == '*/15':
            return 'toStartOfFifteenMinutes'
        elif h == '*/1':
            return 'toStartOfHour'
        elif d == '*/1':
            return 'toStartOfDay'
        elif d == '*/4':
            return 'toStartOfQuarter'
        elif M == '*/1':
            return 'toStartOfMonth'
        elif y == '*/1':
            return 'toStartOfYear'
        else:
            logger.info('[getToStartOf ELSE] crontab: {} => return default value: toStartOfHour'.format(crontab))
            return 'toStartOfHour'
    except:
        logger.error('[getToStartOf ERROR] crontab: {} => return default value: toStartOfHour'.format(crontab))
        return 'toStartOfHour'

def getDelta(delta):
    try:
        unit, num = delta.split('=')[0], int(delta.split('=')[1])
        if unit == 'seconds':
            return datetime.timedelta(seconds=num)
        elif unit == 'minutes':
            return datetime.timedelta(minutes=num)
        elif unit == 'hours':
            return datetime.timedelta(hours=num)
        elif unit == 'days':
            return datetime.timedelta(days=num)
        elif unit == 'weeks':
            return datetime.timedelta(weeks=num)
        else:
            logger.error('[getDelta ELSE] delta: {} => return default value: days=1'.format(delta))
            return datetime.timedelta(days=1)
    except:
        logger.error('[getDelta ERROR] delta: {} => return default value: days=1'.format(delta))
        return datetime.timedelta(days=1)

def attack_sql(param, limit=10, att_limit=10):
    main = """with
                --replaceRegexpAll(lower(http_host), '[0-9]', 'A') as http_host2
                replaceAll(lower(http_host), '.', ' ') as http_host_keyword_arr

                --, replaceRegexpAll(lower(http_path), '[0-9]', 'A') as http_path2
                , replaceRegexpAll(lower(http_path), '[/.,/-?!@#$%^&/_]', ' ') as http_path_keyword_arr
                --, splitByChar('.', http_path3) as http_path_keyword_arr

                --, replaceRegexpAll(lower(http_query), '[0-9]', 'A') as http_query2
                , replaceRegexpAll(lower(http_query), '[/.0-9,=:/?!@#$%^&/_]', ' ') as http_query_keyword_arr
                --, splitByChar('.', http_query3) as http_query_keyword_arr

                --, replaceRegexpAll(lower(http_agent), '[0-9]', 'A') as http_agent2
                , replaceRegexpAll(lower(http_agent), '[/. ,/-?!@#$%^&/_()]', ' ') as http_agent_keyword_arr
        select toStartOfHour(logtime) as lgtime, max(end_time), src_ip, dst_ip,
                length(toSting(count())) as cnt,
                avg(if(extract(http_method, 'GET')=='GET', 1, 0)) as GET,
                avg(if(extract(http_method, 'HEAD')=='HEAD', 1, 0)) as HEAD,
                avg(if(extract(http_method, 'PUT')=='PUT', 1, 0)) as PUT,
                avg(if(extract(http_method, 'POST')=='POST', 1, 0)) as POST,
                avg(if(extract(http_method, 'OPTIONS')=='OPTIONS', 1, 0)) as OPTIONS,
                avg(if(extract(http_method, 'DELETE')=='DELETE', 1, 0)) as DELETE,
                avg(if(extract(http_method, 'TRACE')=='TRACE', 1, 0)) as TRACE,
                avg(if(extract(http_method, 'CONNECT')=='CONNECT', 1, 0)) as CONNECT,
                avg(if(extract(http_method, 'PATCH')=='PATCH', 1, 0)) as PATCH,
                avg(if(extract(http_method, 'REPORT')=='REPORT', 1, 0)) as REPORT,
                avg(if(extract(http_method, 'PROPFIND')=='PROPFIND', 1, 0)) as PROPFIND,
                avg(if(http_method=='-', 1, 0)) as OTH,

                uniqExact(dst_port) as uniq_dst_port,
                entropy(dst_port) as port_entropy,
                avg(length(toString(dst_port))==1) as port_1,
                avg(length(toString(dst_port))==2) as port_2,
                avg(length(toString(dst_port))==3) as port_3,
                avg(length(toString(dst_port))==4) as port_4,
                avg(length(toString(dst_port))==5) as port_5,

                --avg(if(extract(http_retcode, '10')=='10', 1, 0)) as info_st,
                --avg(if(extract(http_retcode, '20')=='20', 1, 0)) as succ_st,
                --avg(if(extract(http_retcode, '30')=='30', 1, 0)) as redir_st,
                --avg(if(extract(http_retcode, '40')=='40', 1, 0)) as cler_st,
                --avg(if(extract(http_retcode, '50')=='50', 1, 0)) as serer_st,
                --avg(if(http_retcode=='-', 1, 0)) as oth_st,

                entropy(http_path_keyword_arr) as path_ent,
                entropy(http_query_keyword_arr) as query_ent,
                entropy(http_agent_keyword_arr) as agent_ent,
                length(groupUniqArray(http_path_keyword_arr)) as path_len,
                length(groupUniqArray(http_query_keyword_arr)) as query_len,
                length(groupUniqArray(http_agent_keyword_arr)) as agent_len,
                avg(length(http_path_keyword_arr)) as path_uni,
                avg(length(http_query_keyword_arr)) as query_uni,
                avg(length(http_agent_keyword_arr)) as agent_uni,

                --groupArray(arrayStringConcat(extractAll(http_host, '[a-zA-Z.:,=&?$/]'))) as http_host1,
                --groupArray(arrayStringConcat(extractAll(http_path, '[a-zA-Z.:,=&?$/]'))) as http_path1,
                --groupArray(arrayStringConcat(extractAll(http_query, '[a-zA-Z.:,=&?$/]'))) as http_query1,
                --groupArray(arrayStringConcat(extractAll(http_agent, '[a-zA-Z.:,=&?$/]'))) as http_agent1,

                --arrayDistinct(groupUniqArray(http_host_keyword_arr)) as _http_host_keyword
                arrayDistinct(groupUniqArray(http_paht_keyword_arr)) as _http_path_keyword
                , arrayDistinct(groupUniqArray(http_query_keyword_arr)) as _http_query_keyword
                , arrayDistinct(groupUniqArray(http_agent_keyword_arr)) as _http_agent_keyword,
                
                """

    beaconing = """1
                from dti_qm_httpd
                where logtime >= toDateTime(toDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                    --logtime >= '2020-01-15 00:00:00 and logtime <= '2020-01-15 00:59:59'
                and
                (
                    (lower(http_path) like '%cgi-%')
                    or (lower(http_path) like '%\.php') -- new Vulnerability
                    or (lower(http_query) like '%..%/%..%/%') -- file download
                    or (lower(http_path) like '%..%2f%')
                    or (lower(http_path) like '%..%255c%')
                    or (lower(http_path) like '%..%252f%') -- file download
                    or (lower(http_query) like '%etc%passwd%')
                    or (lower(http_query) like '%etc%shadow%')
                    or (lower(http_query) like '%etc%hosts%')
                    or (lower(http_query) like '%etc%shells%')
                    or (lower(http_path) like '%php%admin%')
                    or (lower(http_path) like '%..%255c%')
                    or (lower(http_path) like '%..%252f%')
                    or (lower(http_query) like '%wp-includes%')
                    or (lower(http_query) like '%wp-admin%')
                    or (lower(http_query) like '%wp-content%')
                    or (lower(http_query) like '%ftp://%')
                    or (lower(http_query) like '%.hta%')
                )

                and http_path != 'rcmd.shinhan.com/so/recommend'
                and http_path != 'sol.shinhan.com/api/superPlatform/web/account/list/all.sp'
                and (http_query not like '%CODEGUARD_CMD%')
                and http_path != 'sol.shinhan.com/CodeGuard/check.jsp'
                and (http_path not like '%doit5.com%')
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )

            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    injection = """2
                from dti_qm_httpd
                where logtime >= toDateTime(toDate('{logdate_m}')-31) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                and (
                    (extractAll(lower(http_query),'(information_schema|table_name|order\s+by|file\_get\_contents|\%c0\%ae|require\_once|declare|exec.+master|navigateandfind|procedure|procedure|xp_cmdshell|passthru|benchmark|sessionstorage|postmessage|constructor|setimmediate|settimeout|load\_file|concat|union.*select|ls \-[a-z]*|nc -[lvp]*|rm.*\-rf|\%00|\%zz|ysaccessobjects|msysaces|msysobjects|msysqueries|msysrelationships|msysaccessstorage|msysaccessxml|msysmodules|msysmodules2|aster\.\.sysdatabases|ysql\.db|\.database\_name|qlite(\_temp)?\_master|pg\_(catalog|toast)|northwind|tempdb|nslookup|sysdate\)') != [])
                    )
                and http_path != 'rcmd.shinhan.com/so/recommend'
                and http_path != 'sol.shinhan.com/api/superPlatform/web/account/list/all.sp'
                and (http_query not like '%CODEGUARD_CMD%')
                and http_path != 'sol.shinhan.com/CodeGuard/check.jsp'
                and (http_path not like '%doit5.com%')
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    spoofing = """3
                from dti_qm_httpd
                where logtime >= toDateTime(toDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+3)
                and http_path != 'rcmd.shinhan.com/so/recommend'
                and http_path != 'sol.shinhan.com/api/superPlatform/web/account/list/all.sp'
                and (http_query not like '%CODEGUARD_CMD%')
                and http_path != 'sol.shinhan.com/CodeGuard/check.jsp'
                and (http_path not like '%doit5.com%')
                and http_xfor != src_ip
                and http_xfor != '-'
                and http_xfor not like '%,%'
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            having count(src_ip) >= {att_lim} + 10
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    scanning = """4
                from dti_qm_httpd
                where logtime >= toDateTime(toDate('{logdate_m}')-31) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                and (extractAll(lower(http_agent), 'metasploit|nmap|shodan|pharos|botnet|owasp|openvas|nessus') != [])
                and http_path != 'rcmd.shinhan.com/so/recommend'
                and http_path != 'sol.shinhan.com/api/superPlatform/web/account/list/all.sp'
                and (http_query not like '%CODEGUARD_CMD%')
                and http_path != 'sol.shinhan.com/CodeGuard/check.jsp'
                and (http_path not like '%doit5.com%')
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    credential = """5
                from dti_qm_httpd
                where logtime >= toDateTime(toDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+2)
                --logtime >= '2020-04-22 15:00:00' and logtime < '2020:04:23 15:00:00'
                and (extractAll(lower(http_query), '(user_pass|user_login|user_id|userid%passwd)') != [])
                and http_path != 'rcmd.shinhan.com/so/recommend'
                and http_path != 'sol.shinhan.com/api/superPlatform/web/account/list/all.sp'
                and (http_query not like '%CODEGUARD_CMD%')
                and http_path != 'sol.shinhan.com/CodeGuard/check.jsp'
                and (http_path not like '%doit5.com%')
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    xss = """6
                from dti_qm_httpd
                where logtime >= toDateTime(toDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                and 
                (
                    (extractAll(lower(http_query), '<+[a-z0-9\\~\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^\\_\\`\\~]*>') != [])
                )
                and http_path != 'rcmd.shinhan.com/so/recommend'
                and http_path != 'sol.shinhan.com/api/superPlatform/web/account/list/all.sp'
                and (http_query not like '%CODEGUARD_CMD%')
                and http_path != 'sol.shinhan.com/CodeGuard/check.jsp'
                and (http_path not like '%doit5.com%')
                and http_query != '-'
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    return [main+beaconing, main+injection, main+spoofing, main+scanning, main+credential, main+xss]
#   return [main_beaconing, main+xss]

def hijack_sql(param, limit=10, att_limit=10):
    sql = """
        select toStartOfHour(logime) as lgtime, max(end_time), src_ip, --dst_ip,
                count() as cnt,

                avg(if(extract(http_method, 'GET')=='GET', 1, 0)) as GET,
                avg(if(extract(http_method, 'HEAD')=='HEAD', 1, 0)) as HEAD,
                avg(if(extract(http_method, 'PUT')=='PUT', 1, 0)) as PUT,
                avg(if(extract(http_method, 'POST')=='POST', 1, 0)) as POST,
                avg(if(extract(http_method, 'OPTIONS')=='OPTIONS', 1, 0)) as OPTIONS,
                avg(if(extract(http_method, 'DELETE')=='DELETE', 1, 0)) as DELETE,
                avg(if(extract(http_method, 'TRACE')=='TRACE', 1, 0)) as TRACE,
                avg(if(extract(http_method, 'CONNECT')=='CONNECT', 1, 0)) as CONNECT,
                avg(if(extract(http_method, 'PATCH')=='PATCH', 1, 0)) as PATCH,
                avg(if(extract(http_method, 'REPORT')=='REPORT', 1, 0)) as REPORT,
                avg(if(extract(http_method, 'PROPFIND')=='PROPFIND', 1, 0)) as PROPFIND,
                avg(if(http_method=='-', 1, 0)) as OTH,

                avg(bytes_forward) as req_bmean,
                varPop(bytes_forward) as req_bvar,
                avg(bytes_backward) as resp_bmean,
                varPop(bytes_backward) as resp_bvar,
                avg(packets_forward) as req_pmean,
                varPop(packets_forward) as req_pvar,
                avg(packets_backward) as resp_pmean,
                varPop(packets_backward) as resp_pvar,

                avg(if(extract(http_retcode, '10')=='10', 1, 0)) as info_st,
                avg(if(extract(http_retcode, '20')=='20', 1, 0)) as succ_st,
                avg(if(extract(http_retcode, '30')=='30', 1, 0)) as redir_st,
                avg(if(extract(http_retcode, '40')=='40', 1, 0)) as cler_st,
                avg(if(extract(http_retcode, '50')=='50', 1, 0)) as serer_st,
                avg(if(http_retcode=='-', 1, 0)) as oth_st,

                --groupArray(arrayStringConcat(extractAll(http_host, '[a-zA-Z.:,=&?$/]'))) as http_host,
                groupArray(arrayStringConcat(extractAll(http_path, '[a-zA-Z.:,=&?$/]'))) as http_pat,
                groupArray(arrayStringConcat(extractAll(http_query, '[a-zA-Z.:,=&?$/]'))) as http_query,
                groupArray(arrayStringConcat(extractAll(http_cookie, '[a-zA-Z.:,=&?$/]'))) as http_cooki,
                1

                from dti_qm_httpd
                where logtime >= '{logdate_m}'
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
                and http_cookie like '%SESSION%'

                group by src_ip, lgtime
                having count() > 200
                and entropy(http_path) == 0
                limit {lim}""".format(logdate_m=param['logtime_e'], lim=limit, att_lim=att_limit)

    return sql


def attack_sql2(param, limit=10, att_limit=10):
    main = """
                select toStartOfHour(logtime) as lgtime, max(end_time), src_ip, dst_ip,
                
                count() as cnt,

                avg(if(extract(http_method, 'GET')=='GET', 1, 0)) as GET,
                avg(if(extract(http_method, 'HEAD')=='HEAD', 1, 0)) as HEAD,
                avg(if(extract(http_method, 'PUT')=='PUT', 1, 0)) as PUT,
                avg(if(extract(http_method, 'POST')=='POST', 1, 0)) as POST,
                avg(if(extract(http_method, 'OPTIONS')=='OPTIONS', 1, 0)) as OPTIONS,
                avg(if(extract(http_method, 'DELETE')=='DELETE', 1, 0)) as DELETE,
                avg(if(extract(http_method, 'TRACE')=='TRACE', 1, 0)) as TRACE,
                avg(if(extract(http_method, 'CONNECT')=='CONNECT', 1, 0)) as CONNECT,
                avg(if(extract(http_method, 'PATCH')=='PATCH', 1, 0)) as PATCH,
                avg(if(extract(http_method, 'REPORT')=='REPORT', 1, 0)) as REPORT,
                avg(if(extract(http_method, 'PROPFIND')=='PROPFIND', 1, 0)) as PROPFIND,
                avg(if(http_method=='-', 1, 0)) as OTH,

                avg(bytes_forward) as req_bmean,
                varPop(bytes_forward) as req_bvar,
                avg(bytes_backward) as resp_bmean,
                varPop(bytes_backward) as resp_bvar,
                avg(packets_forward) as req_pmean,
                varPop(packets_forward) as req_pvar,
                avg(packets_backward) as resp_pmean,
                varPop(packets_backward) as resp_pvar,

                avg(if(extract(http_retcode, '10')=='10', 1, 0)) as info_st,
                avg(if(extract(http_retcode, '20')=='20', 1, 0)) as succ_st,
                avg(if(extract(http_retcode, '30')=='30', 1, 0)) as redir_st,
                avg(if(extract(http_retcode, '40')=='40', 1, 0)) as cler_st,
                avg(if(extract(http_retcode, '50')=='50', 1, 0)) as serer_st,
                avg(if(http_retcode=='-', 1, 0)) as oth_st,

                groupArray(arrayStringConcat(extractAll(http_host, '[a-zA-Z.:,=&?$/]'))) as http_host,
                groupArray(arrayStringConcat(extractAll(http_path, '[a-zA-Z.:,=&?$/]'))) as http_path,
                groupArray(arrayStringConcat(extractAll(http_query, '[a-zA-Z.:,=&?$/]'))) as http_query,

                """

    beaconing = """1
                from dti_qm_http
                where logtime >= toDateTime(toDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logtime_m}')+31)
                and src_ip global in
                (
                    select distinct src_ip
                    from
                    (
                        select src_ip
                        from dti_blockd
                        where attack_name like '%attempt%'
                        or attack_name like '%Vulnerability%'
                        and attack_name not like '%SQL%'
                        and attack_name not like '%Scan%'

                        group by src_ip
                        having count(src_ip) >= {att_lim}+150
                    )all full outer join
                    (
                        select src_ip
                        from dti_wafd
                        where attack_name like '%attempt%'
                        or attack_name like '%Vulnerability%'
                        and attack_name not like '%SQL%'
                        and attack_name not like '%Scan%'
                        group by src_ip
                        having count(src_ip) >= {att_lim}+150
                    )using src_ip
                )
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )

            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    injection = """2
                from dti_qm_httpd
                where logtime >= toDateTime(taDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                and src_ip global in
                (
                    select src_ip
                    from
                    (
                        select src_ip
                        from dti_blockd
                        where attack_name like '%SQL%'
                        group by src_ip
                        having count(src_ip) >= {att_lim}+50
                    )all full outer join
                    (
                        select src_ip
                        from dti_wafd
                        where attack_name like '%Inject%'
                        group by src_ip
                        having count(src_ip) >= {att_lim}+50
                    )using src_ip
                )
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    spoofing = """3
                from dti_qm_httpd
                where logtime >= toDateTime(taDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                and http_xfor != src_ip
                and http_xfor != '-'
                and http_xfor not like '%,%'
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            having count(src_ip) >= {att_lim} + 10
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    scanning = """4
                from dti_qm_httpd
                where logtime >= toDateTime(taDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                and src_ip global in
                (
                    select src_ip
                    from
                    (
                        select src_ip
                        from dti_blockd
                        where attack_name like '%Scan%'
                        group by src_ip
                        having count(src_ip) >= {att_lim}+500
                    )all full outer join
                    (
                        select src_ip
                        from dti_wafd
                        where attack_name like '%Scan%'
                        group by src_ip
                        having count(src_ip) >= {att_lim}+500
                    )using src_ip
                )
                and src_ip gloaal not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    credential = """5
                from dti_qm_httpd
                where logtime >= toDateTime(taDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                and src_ip global in
                (
                    select src_ip
                    from dti_blockd
                    where attack_name like '%HTTP%Brute Force%'
                    group by src_ip
                    having count(src_ip) >= {att_lim}+5

                )
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    xss = """6
                from dti_am_httpd
                where logtime >= toDateTime(taDate('{logdate_m}')) and logtime <= toDateTime(toDate('{logdate_m}')+31)
                and src_ip global in
                (
                    select src_ip
                    from
                    (
                        select src_ip
                        from dti_blockd
                        where attack_name like '%XSS%'
                        group by src_ip
                        having count(src_ip) >= {att_lim}+40
                    )all full outer join
                    (
                        select src_ip
                        from dti_wafd
                        where attack_name like '%XSS%'
                        group by src_ip
                        having count(src_ip) >= {att_lim}+40
                    )using src_ip
                )
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
            group by src_ip, dst_ip, lgtime
            limit {lim}""".format(logdate_m=param['logdate_m'], lim=limit, att_lim=att_limit)
    return [main+beaconing, main+injection, main+spoofing, main+scanning, main+credential, main+xss]
#   return [main_beaconing, main+xss]

def hijack_sql(param, limit=10, att_limit=10):
    sql = """
        select toStartOfHour(logime) as lgtime, max(end_time), src_ip, --dst_ip,
                count() as cnt,

                avg(if(extract(http_method, 'GET')=='GET', 1, 0)) as GET,
                avg(if(extract(http_method, 'HEAD')=='HEAD', 1, 0)) as HEAD,
                avg(if(extract(http_method, 'PUT')=='PUT', 1, 0)) as PUT,
                avg(if(extract(http_method, 'POST')=='POST', 1, 0)) as POST,
                avg(if(extract(http_method, 'OPTIONS')=='OPTIONS', 1, 0)) as OPTIONS,
                avg(if(extract(http_method, 'DELETE')=='DELETE', 1, 0)) as DELETE,
                avg(if(extract(http_method, 'TRACE')=='TRACE', 1, 0)) as TRACE,
                avg(if(extract(http_method, 'CONNECT')=='CONNECT', 1, 0)) as CONNECT,
                avg(if(extract(http_method, 'PATCH')=='PATCH', 1, 0)) as PATCH,
                avg(if(extract(http_method, 'REPORT')=='REPORT', 1, 0)) as REPORT,
                avg(if(extract(http_method, 'PROPFIND')=='PROPFIND', 1, 0)) as PROPFIND,
                avg(if(http_method=='-', 1, 0)) as OTH,

                avg(bytes_forward) as req_bmean,
                varPop(bytes_forward) as req_bvar,
                avg(bytes_backward) as resp_bmean,
                varPop(bytes_backward) as resp_bvar,
                avg(packets_forward) as req_pmean,
                varPop(packets_forward) as req_pvar,
                avg(packets_backward) as resp_pmean,
                varPop(packets_backward) as resp_pvar,

                avg(if(extract(http_retcode, '10')=='10', 1, 0)) as info_st,
                avg(if(extract(http_retcode, '20')=='20', 1, 0)) as succ_st,
                avg(if(extract(http_retcode, '30')=='30', 1, 0)) as redir_st,
                avg(if(extract(http_retcode, '40')=='40', 1, 0)) as cler_st,
                avg(if(extract(http_retcode, '50')=='50', 1, 0)) as serer_st,
                avg(if(http_retcode=='-', 1, 0)) as oth_st,

                --groupArray(arrayStringConcat(extractAll(http_host, '[a-zA-Z.:,=&?$/]'))) as http_host,
                groupArray(arrayStringConcat(extractAll(http_path, '[a-zA-Z.:,=&?$/]'))) as http_pat,
                groupArray(arrayStringConcat(extractAll(http_query, '[a-zA-Z.:,=&?$/]'))) as http_query,
                groupArray(arrayStringConcat(extractAll(http_cookie, '[a-zA-Z.:,=&?$/]'))) as http_cooki,
                1

                from dti_qm_httpd
                where logtime >= '{logdate_m}'
                and src_ip global not in
                (
                    select distinct IP
                    from map_all_ipd
                )
                and http_cookie like '%SESSION%'

                group by src_ip, lgtime
                having count() > 200
                and entropy(http_path) == 0
                limit {lim}""".format(logdate_m=param['logtime_e'], lim=limit, att_lim=att_limit)

    return sql

def eflog_data(param, ips):
    sql = """
            select client_ip, a_s_ratio, h_s_ratio, other_s_ratio, a_f_ratio, h_f_ratio, other_f_ratio, if(diff_length == 0, 0, diff_avg) as diff_avg, if(diff_length == 0, 0, diff_std) as diff_std
            from
            (
                with 
                groupArray(value) as time_sorted,
                arrayEnumerate(time_sorted) as indexes,
                arrayMap(i -> time_sorted[i] - time_sorted[i-1], indexes) as running_diffs

                select client_ip, count() as cnt, running_diffs, arrayPopFront(running_diffs) as running_diffs2, length(running_diffs2) as diff_length, if(empty(running_diffs2)==1, array(0), running_diffs2) as running_diffs3, arraySum(running_diffs3)/diff_length as diff_avg,
                    arrayMap(a -> pow(a-diff_avg, 2), running_diffs3) as diff_diff, round(pow(arraySum(diff_diff)/diff_length, 0.5), 2) as diff_std
                from
                (
                    select client_ip, logtime as value
                    from
                    (
                        select *
                        from dti_eflogd
                        where logdate == '{logdate_day}'
                        and client_ip in {client_ips}
                        limit 100000
                    )
                    order by client_ip, logtime
                )
                group by client_ip
            )
            any left JOIN
            (
                select client_ip, count() as eflog_total_cnt, countIf(service_code=='A1000') as a_service_total, countIf(service_code=='H1001') as h_service_total, countIf((service_code!='H1001')and(service_code!='A1000')) as other_service_total,
                    countIf((service_code=='A1000') and (result==0)) as a_service_sucess, countIf((service_code=='H1001') and (result==0)) as h_service_sucess, countIf((service_code!='H1001')and(service_code!='A1000') and (result==0)) as other_service_sucess, countIf(result==0) as sucess,
                    (a_service_sucess/eflog_total_cnt) as a_s_ratio, (h_service_sucess/eflog_total_cnt) as h_s_ratio, (other_service_sucess/eflog_total_cnt) as other_s_ratio,
                    ((a_service_total - a_service_sucess)/eflog_total_cnt) as a_f_ratio, ((h_service_total - h_service_sucess)/eflog_total_cnt) as h_f_ratio, ((other_service_total - other_service_sucess)/eflog_total_cnt) as other_f_ratio

                from
                (
                    select *
                    from dti_eflogd
                    where logdate == '{logdate_day}'
                    and client_ip in {client_ips}
                    limit 100000
                )
            group by client_ip
            )using client_ip
        """.format(logdate_day=param['logdate_day'], client_ips=ips)
    return sql

def eflog_login(param, ips):
    if len(ips) == 1:
        ips = ips + ips
    sql = """
        select client_ip, sum(if(service_code=='H0015' AND result==0, 1, 0)) as H0015_0,
                sum(if(service_code=='H0015' AND result==1, 1, 0)) as H0015_1,
                sum(if(service_code=='H0020' AND result==0, 1, 0)) as H0020_0,
                sum(if(service_code=='H0020' AND result==1, 1, 0)) as H0020_1,
                sum(if(service_code=='H1001' AND result==0, 1, 0)) as H1001_0,
                sum(if(service_code=='H1001' AND result==1, 1, 0)) as H1001_1,
                sum(if(service_code=='H5010' AND result==0, 1, 0)) as H5010_0,
                sum(if(service_code=='H5010' AND result==1, 1, 0)) as H5010_1,
                sum(if(service_code=='A1000' AND result==0, 1, 0)) as A1000_0,
                sum(if(service_code=='A1000' AND result==1, 1, 0)) as A1000_1
        from dti_dflogd
        where (logtime between toDateTime(toDate('{logdate_day}') - 7) and toDateTime(toDate('{logdate_day}')))
        and client_ip in ({client_ips})
        group by client_ip
        """.format(logdate_day=param['logdate_day'], client_ips=ips)
    return sql

def ast_ip(ips):
    if len(ips) == 1:
        ips = ips + ips
    sql = """
        select IP, 1
        from default.map_IP_AST
        where IP in {client_ips}
        group by IP
        """.format(client_ips=ips)
    return sql

def feedback_sql(m_ip):
    sql = """
        select src_ip, toString(start_time), toString(end_time), result, feedback
        from dti_ai_feedbackd
        where model_id == toString({mid})
        and feedback != -1
        """.format(mid=m_id)
    
    return sql

def http_feedback(ip, start, end):
    sql = """
        select toStartOfHour(logtime) as lgtime, max(end_time), src_ip,
            count() as cnt, 

            avg(if(extract(http_method, 'GET')=='GET', 1, 0)) as GET,
            avg(if(extract(http_method, 'HEAD')=='HEAD', 1, 0)) as HEAD,
            avg(if(extract(http_method, 'PUT')=='PUT', 1, 0)) as PUT,
            avg(if(extract(http_method, 'POST')=='POST', 1, 0)) as POST,
            avg(if(extract(http_method, 'OPTIONS')=='OPTIONS', 1, 0)) as OPTIONS,
            avg(if(extract(http_method, 'DELETE')=='DELETE', 1, 0)) as DELETE,
            avg(if(extract(http_method, 'TRACE')=='TRACE', 1, 0)) as TRACE,
            avg(if(extract(http_method, 'CONNECT')=='CONNECT', 1, 0)) as CONNECT,
            avg(if(extract(http_method, 'PATCH')=='PATCH', 1, 0)) as PATCH,
            avg(if(extract(http_method, 'REPORT')=='REPORT', 1, 0)) as REPORT,
            avg(if(extract(http_method, 'PROPFIND')=='PROPFIND', 1, 0)) as PROPFIND,
            avg(if(http_method=='-', 1, 0)) as OTH,

            avg(bytes_forward) as req_bmean,
            varPop(bytes_forward) as req_bvar,
            avg(bytes_backward) as resp_bmean,
            varPop(bytes_backward) as resp_bvar,
            avg(packets_forward) as req_pmean,
            varPop(packets_forward) as req_pvar,
            avg(packets_backward) as resp_pmean,
            varPop(packets_backward) as resp_pvar,

            avg(if(extract(http_retcode, '10')=='10', 1, 0)) as info_st,
            avg(if(extract(http_retcode, '20')=='20', 1, 0)) as succ_st,
            avg(if(extract(http_retcode, '30')=='30', 1, 0)) as redir_st,
            avg(if(extract(http_retcode, '40')=='40', 1, 0)) as cler_st,
            avg(if(extract(http_retcode, '50')=='50', 1, 0)) as serer_st,
            avg(if(http_retcode=='-', 1, 0)) as oth_st,

            groupArray(arrayStringConcat(extractAll(http_host, '[a-zA-Z.:,=&?$/]'))) as http_host,
            groupArray(arrayStringConcat(extractAll(http_path, '[a-zA-Z.:,=&?$/]'))) as http_pat,
            groupArray(arrayStringConcat(extractAll(http_query, '[a-zA-Z.:,=&?$/]'))) as http_query

            from dti_qm_httpd
            where logtime >= '{start_time}' and end_time <= '{end_time}' and src_ip == '{src_ip}'
            group by src_ip, lgtime
        """.format(start_time=start, end_time=end, src_ip=ip)
    return sql

def nbad_feedback(ip, start, end):
    sql = """
        select toStartOfHour(logtime) as lgtime, max(end_time), src_ip, --dst_ip,
            count() as cnt,

            avg(if(extract(http_method, 'GET')=='GET', 1, 0)) as GET,
            avg(if(extract(http_method, 'HEAD')=='HEAD', 1, 0)) as HEAD,
            avg(if(extract(http_method, 'PUT')=='PUT', 1, 0)) as PUT,
            avg(if(extract(http_method, 'POST')=='POST', 1, 0)) as POST,
            avg(if(extract(http_method, 'OPTIONS')=='OPTIONS', 1, 0)) as OPTIONS,
            avg(if(extract(http_method, 'DELETE')=='DELETE', 1, 0)) as DELETE,
            avg(if(extract(http_method, 'TRACE')=='TRACE', 1, 0)) as TRACE,
            avg(if(extract(http_method, 'CONNECT')=='CONNECT', 1, 0)) as CONNECT,
            avg(if(extract(http_method, 'PATCH')=='PATCH', 1, 0)) as PATCH,
            avg(if(extract(http_method, 'REPORT')=='REPORT', 1, 0)) as REPORT,
            avg(if(extract(http_method, 'PROPFIND')=='PROPFIND', 1, 0)) as PROPFIND,
            avg(if(http_method=='-', 1, 0)) as OTH,

            avg(if(extract(http_retcode, '10')=='10', 1, 0)) as info_st,
            avg(if(extract(http_retcode, '20')=='20', 1, 0)) as succ_st,
            avg(if(extract(http_retcode, '30')=='30', 1, 0)) as redir_st,
            avg(if(extract(http_retcode, '40')=='40', 1, 0)) as cler_st,
            avg(if(extract(http_retcode, '50')=='50', 1, 0)) as serer_st,
            avg(if(http_retcode=='-', 1, 0)) as oth_st,

            groupArray(arrayStringConcat(extractAll(http_host, '[a-zA-Z.:,=&?$/]'))) as http_host,
            groupArray(arrayStringConcat(extractAll(http_path, '[a-zA-Z.:,=&?$/]'))) as http_pat,
            groupArray(arrayStringConcat(extractAll(http_query, '[a-zA-Z.:,=&?$/]'))) as http_query
            
            from dti_qm_httpd
            where logtime >= '{start_time}' and end_time <= '{end_time}' and src_ip == '{src_ip}'
            group by src_ip, lgtime
        """.format(start_time=start, end_time=end, src_ip=ip)
    return sql

def dns_feedback(ip, start, end):
    sql = """
        select lgtime, end_time, src_ip, cnt,
            hour_0, hour_1, hour_2, hour_3, hour_4, hour_5, hour_6, hour_7, hour_8, hour_9, hour_10,
            hour_11, hour_12, hour_13, hour_14, hour_15, hour_16, hour_17, hour_18, hour_19, hour_20, hour_21, hour_22, hour_23,
            day_1, day_2, day_3, day_4, day_5, day_6, day_7, global_osint, global_kfisac, dns_query, dns_answer

        from
        (
            select toStartOfHour(logtime) as lgtime, max(end_time) as end_time, src_ip,
                count() as cnt,
                groupArray(arrayStringConcat(extractAll(dns_query, '[a-zA-Z.:,=&?$/]'))) as dns_query,
                groupArray(arrayStringConcat(extractAll(dns_answer, '[a-zA-Z.:,=&?$/]'))) as dns_answer,
                avg(global_osint) as global_osint, avg(global_kfisac) as global_kfisac, IPv4NumToStringClassC(IPv4StringToNum(src_ip)) as ip_range
            from
            (
                select logtime, end_time, src_ip, bytes_forward, bytes_backward, 
                    packets_forward, packets_backward, dns_query, dns_answer, global_osint
                from
                (    
                    select logtime, end_time, src_ip, bytes_forward, bytes_backward, 
                        packets_forward, packets_backward, dns_query, dns_answer
                    from dti_qm_dnsd
                    where logtime >= '{start_time}' and end_time <= '{end_time}' and src_ip == '{src_ip}'
            
                )any left join
                (
                    select target as dns_query, 1 as gloabl_osint
                    from osint
                )using dns_query
            )any left join
            (
                select IPv4NumToStringClassC(value_num_start) as ip_range, 1 as global_kfisac
                from gi_kfisac
            )using ip_range
            group by src_ip, lgtime
            having lgtime == '{start_time}' and end_time == '{end_time}' and src_ip == '{src_ip}'
        ) any left join
        (
            select src_ip,
                avg(if(toHour(logtime)==0, 1, 0)) as hour_0,
                avg(if(toHour(logtime)==1, 1, 0)) as hour_1,
                avg(if(toHour(logtime)==2, 1, 0)) as hour_2,
                avg(if(toHour(logtime)==3, 1, 0)) as hour_3,
                avg(if(toHour(logtime)==4, 1, 0)) as hour_4,
                avg(if(toHour(logtime)==5, 1, 0)) as hour_5,
                avg(if(toHour(logtime)==6, 1, 0)) as hour_6,
                avg(if(toHour(logtime)==7, 1, 0)) as hour_7,
                avg(if(toHour(logtime)==8, 1, 0)) as hour_8,
                avg(if(toHour(logtime)==9, 1, 0)) as hour_9,
                avg(if(toHour(logtime)==10, 1, 0)) as hour_10,
                avg(if(toHour(logtime)==11, 1, 0)) as hour_11,
                avg(if(toHour(logtime)==12, 1, 0)) as hour_12,
                avg(if(toHour(logtime)==13, 1, 0)) as hour_13,
                avg(if(toHour(logtime)==14, 1, 0)) as hour_14,
                avg(if(toHour(logtime)==15, 1, 0)) as hour_15,
                avg(if(toHour(logtime)==16, 1, 0)) as hour_16,
                avg(if(toHour(logtime)==17, 1, 0)) as hour_17,
                avg(if(toHour(logtime)==18, 1, 0)) as hour_18,
                avg(if(toHour(logtime)==19, 1, 0)) as hour_19,
                avg(if(toHour(logtime)==20, 1, 0)) as hour_20,
                avg(if(toHour(logtime)==21, 1, 0)) as hour_21,
                avg(if(toHour(logtime)==22, 1, 0)) as hour_22,
                avg(if(toHour(logtime)==23, 1, 0)) as hour_23,
                avg(if(toDayOfWeek(logtime)==1, 1, 0)) as day_1,
                avg(if(toDayOfWeek(logtime)==2, 1, 0)) as day_2,
                avg(if(toDayOfWeek(logtime)==3, 1, 0)) as day_3,
                avg(if(toDayOfWeek(logtime)==4, 1, 0)) as day_4,
                avg(if(toDayOfWeek(logtime)==5, 1, 0)) as day_5,
                avg(if(toDayOfWeek(logtime)==6, 1, 0)) as day_6,
                avg(if(toDayOfWeek(logtime)==7, 1, 0)) as day_7

            from dti_qm_dnsd
            where toStartOfHour(logtime) == '{start_time}' and src_ip == '{src_ip}'
            group by src_ip
            having max(end_time) == '{end_time}'
        ) using src_ip
        """.format(start_time=start, end_time=end, src_ip=ip)
    return sql

def dns_cluster_sql(param, ips):
    sql = """
        select *
        from(
            select 1 as idx,
                count() nas cnt,
    
                --avg(bytes_forward) as req_bmean,
                --varPop(bytes_forward) as req_bvar,
                --avg(bytes_backward) as resp_bmean,
                --varPop(bytes_backward) as resp_bvar,
                --avg(packets_forward) as req_pmean,
                --varPop(packets_forward) as req_pvar,
                --avg(packets_backward) as resp_pmean,
                --varPop(packets_backward) as resp_pvar,

                avg(if(toHour(logtime)==0, 1, 0)) as hour_0,
                avg(if(toHour(logtime)==1, 1, 0)) as hour_1,
                avg(if(toHour(logtime)==2, 1, 0)) as hour_2,
                avg(if(toHour(logtime)==3, 1, 0)) as hour_3,
                avg(if(toHour(logtime)==4, 1, 0)) as hour_4,
                avg(if(toHour(logtime)==5, 1, 0)) as hour_5,
                avg(if(toHour(logtime)==6, 1, 0)) as hour_6,
                avg(if(toHour(logtime)==7, 1, 0)) as hour_7,
                avg(if(toHour(logtime)==8, 1, 0)) as hour_8,
                avg(if(toHour(logtime)==9, 1, 0)) as hour_9,
                avg(if(toHour(logtime)==10, 1, 0)) as hour_10,
                avg(if(toHour(logtime)==11, 1, 0)) as hour_11,
                avg(if(toHour(logtime)==12, 1, 0)) as hour_12,
                avg(if(toHour(logtime)==13, 1, 0)) as hour_13,
                avg(if(toHour(logtime)==14, 1, 0)) as hour_14,
                avg(if(toHour(logtime)==15, 1, 0)) as hour_15,
                avg(if(toHour(logtime)==16, 1, 0)) as hour_16,
                avg(if(toHour(logtime)==17, 1, 0)) as hour_17,
                avg(if(toHour(logtime)==18, 1, 0)) as hour_18,
                avg(if(toHour(logtime)==19, 1, 0)) as hour_19,
                avg(if(toHour(logtime)==20, 1, 0)) as hour_20,
                avg(if(toHour(logtime)==21, 1, 0)) as hour_21,
                avg(if(toHour(logtime)==22, 1, 0)) as hour_22,
                avg(if(toHour(logtime)==23, 1, 0)) as hour_23,
                avg(if(toDayOfWeek(logtime)==1, 1, 0)) as day_1,
                avg(if(toDayOfWeek(logtime)==2, 1, 0)) as day_2,
                avg(if(toDayOfWeek(logtime)==3, 1, 0)) as day_3,
                avg(if(toDayOfWeek(logtime)==4, 1, 0)) as day_4,
                avg(if(toDayOfWeek(logtime)==5, 1, 0)) as day_5,
                avg(if(toDayOfWeek(logtime)==6, 1, 0)) as day_6,
                avg(if(toDayOfWeek(logtime)==7, 1, 0)) as day_7,
                entropy(dns_query)

            from dti_qm_dnsd
            where (logtime between toDateTime(toDate('{logdate_day}') - 7) and toDateTime(toDate('{logdate_day}')))
                and src_ip in {client_ips}
        )any left join
        (
            select idx, rareness, global_osint, global_kfisac
            from
            (
                select 1 as idx, avg(rareness) as rareness
                from
                (
                    select src_ip, dns_query
                    from dti_qm_dnsd
                    where (logtime between toDateTime(toDate('{logdate_day}') - 7) and toDateTime(toDate('{logdate_day}')))
                        and src_ip in {client_ips}
                )any left join
                (
                    select dns_query, (countEqual(groupArray(dns_query), dns_query)/(select count() from dti_qm_dnsd where (logtime between toDateTime(toDate('{logdate_day}') - 7) and toDateTime(toDate('{logdate_day}'))))) as rareness
                    from dti_qm_dnsd
                    where (logtime between toDateTime(toDate('{logdate_day}') - 7) and toDateTime(toDate('{logdate_day}')))
                    group by dns_query
                )using dns_query
            )any left join
            (
                select 1 as idx, avg(global_osint) as global_osint, avg(global_kfisac) as global_kfisac
                from
                (
                    select avg(global_osint) as global_osint, avg(global_kfisac) as global_kfisac, IPv4NumToStringClassC(IPv4StringToNum(src_ip)) as ip_range
                    from
                    (
                        select src_ip, global_osint
                        from
                        (
                            select src_ip, dns_query
                            from dti_qm_dnsd
                            where (logtime between toDateTime(toDate('{logdate_day}') - 7) and toDateTime(toDate('{logdate_day}')))
                                and src_ip in {client_ips}
                        )any left join
                        (
                            select target as dns_query, 1 as global_osint
                            from osint
                        )using dns_query
                    )any left join
                    (
                        select IPv4NumToStringClassC(value_num_start) as ip_range, 1 as global_kfisac
                        from gi_kfisac
                    )using ip_range
                    group by ip_range
                )
            )using idx
        )using idx
        """.format(logdate_day=param['logdate_day'], client_ips=ips)
    return sql

def rmse(y_pred, y_true):
    return np.sqrt(list(np.mean(np.square(y_pred - y_true), axis = 1)))

def save_obj(obj, name):
    # logger.info('save_obj: {}/obj/{}.pkl'.format(pwd, name))
    with open('{}/obj/{}.pkl'.format(pwd, name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    # logger.info('load_obj: {}/obj/{}.pkl'.format(pwd, name))
    with open('{}/obj/{}.pkl'.format(pwd, name), 'rb') as f:
        return pickle.load(f)

def update_obj(name):
    from_obj = '{}/obj/{}.pkl'.format(pwd, name)
    to_obj = '{}/obj/{}.pkl'.format(pwd, name)

    if os.path.exists(to_obj):
        os.system('rm -rf {}'.format(to_obj))
        # shutil.rmtree(to_obj)

    os.system('mv {} {}'.format(from_obj, to_obj))
    # shutil.move(from_obj, to_obj)
    logger.info('update_obj: {} -> {}'.format(from_obj, to_obj))

def update_model(from_model, to_model):
    if os.path.exists(to_model):
        os.system('rm -rf {}'.format(to_model))
        # shutil.rmtree(to_model)

    os.system('mv {} {}'.format(from_model, to_model))
    # shutil.move(from_model, to_model)
    logger.info('update_model: {} -> {}'.format(from_model, to_model))

##################################################################################

def similarity(samples, feedbacks, threshold=0.8):
    sims = np.diag(cosine_similarity(samples, feedbacks))
    return np.all(np.less(sims, threshold))

def create_sql_data(data_sample):
    sql_keywords = ['union', 'drop', 'insert', 'create', 'merge', 'all', 'select', 'as', 'from', 'table', 'where', 'and', 'or', 'in', 'like', 'limit']
    sql_sample = []

    for i in range(np.random.randint(4,6)):
        len_of_sql = np.random.randint(4, int(len(sql_keywords)))
        sql = np.random.randint(0, len(sql_keywords), len_of_sql)
        sql.sort()
        sql_sample.append(''.join(np.array(sql_keywords)[sql]))

    for i in range(np.random.randint(0, 1)):
        sql_sample.append(data_sample[np.random.randint(0, len(data_sample))])
    random.shuffle(sql_sample)
    return tuple(sql_sample)

####################################################################################
class counter:
    def __init__(self, predict=False, path=None, model_name=None):
        self.predict = predict
        self.model_name = model_name
        self.path_graph = '{}/obj/counters/'.format(path)
        self.dict_file = '/colddata/ctilab/app/ai/model_total/obj/'.format(path, model_name + '_')
        print(self.path_graph)
        print(self.dict_file)
        if not os.path.exists(self.path_graph):
            os.makedirs(self.path_graph)

    def start_count(self, data=None):
        manager = Manager()
        dictionary = {}
        array_dict = {}
        p_list = []
        for i in range(len(data[0])):
            if os.path.exists(self.path_graph + self.model_name + '_' + str(i) + '.pkl'):
                dictionary[i] = manager.dict(load_obj('counters/' + self.model_name + '_' + str(i)))
                print('DICT LOADED')
                if self.predict:
                    array_dict[i] = manager.list()
            else:
                dictionary[i] = manager.dict()
            if not self.predict:
                if isinstance(data[0][i], str):
                    p_list.append(Process(target=self.string_count, args=(data[:, i], dictionary[i], i)))
                else:
                    p_list.append(Process(target=self.value_count, args=(data[:, i], dictionary[i], i)))
            else:
                if isinstance(data[0][i], str):
                    p_list = append(Process(target=self.string_predict, args=(data[:, i], dictionary[i], array_dict[i],)))
                else:
                    p_list = append(Process(target=self.value_predict, args=(data[:, i], dictionary[i], array_dict[i],)))
        
        for p in p_list:
            p.start()

        for p in p_list:
            p.join()

        if not self.predict:
            for d in dictionary.keys():
                print(dictionary[d])
                save_obj(dict(dictionary[d]), 'counters/' + self.model_name + '_' + str(d))
            return dictionary
        else:
            final_result = []
            for d in range(len(array_dict)):
                final_result.append(array_dict[d])
            return final_result

    def string_count(self, col, d, i):
        for i in col:
            for n in i.split():
                if n in d.keys():
                    d[n] += 1
                else:
                    d[n] = 1
        print(i)

    def value_count(self, col, d, i):
        unique, counts = np.unique(col, return_counts=True)
        for c, val in enumerate(unique):
            if val in d.keys():
                d[val] += counts[c]
            else:
                d[val] = counts[c]
        print(i)

    def value_predict(self, col, d, a):
        s = sum(d.values())
        for n in range(len(col)):
            try:
                a.appned(1 - np.sqrt(d[col[n]]/s))
            except:
                a.append(0)

    def string_predict(self, col, d, a):
        s = sum(d.values())
        for i in range(len(col)):
            total = 0
            cnt = 0
            try:
                for n in col[i].split():
                    total += d[n]
                    cnt += 1
                a.append(1 - np.sqrt((total/cnt)/s))
            except:
                a.append(0)


class dir_counter:
    def __init__(self, predict=False):
        self.predict = predict

    def start_count(self, all_data=None, connection=None, path=None, model_name=None):
        uniq = np.unique(connection)
        manager = Manager()
        self.path_graph = {}
        self.dict_file = {}
        self.model_name = model_name + '_train'
        self.total_dict_file = 'counters/{}/dict'.format(self.model_name)
        self.total_dict_path = '{}/obj/counters/{}/'.format(path, self.model_name)
        if not os.path.exists(self.total_dict_path):
            os.makedirs(self.total_dict_path)
        self.total_dict = {}
        for i in range(len(all_data[0])):
            if os.path.exists(self.total_dict_path + 'dict_' + str(i) + '.pkl'):
                self.total_dict[i] = manager.dict(load_obj(self.total_dict_file+'_'+str(i)))
                print('TOTAL_DICT LOADED')
            else:
                self.total_dict[i] = manager.dict()
        
        prediction = [[] for i in range(all_data.shape[1])]
        for u in uniq:
            data = all_data[connection == u]
            self.path_graph[u] = '{}/obj/counters/{}/'.format(path, self.model_name + '_' + u)
            self.dict_file[u] = '/obj/counters/{}/dict'.format(self.model_name + '_' + u)
            if not os.path.exists(self.path_graph[u]):
                os.makedirs(self.path_graph[u])
            dictionary = {}
            array_dict = {}
            p_list = []
            for i in range(len(data[0])):
                if os.path.exists(self.path_graph[u]+'dict_' + str(i) + '.pkl'):
                    print('DICT LOADED ' + 'dict_' + str(i) + '.pkl')
                    dictionary[i] = manager.dict(load_obj('counters/'+self.model_name + '_' + u + '/dict_' + str(i)))
                    if self.predict:
                        array_dict[i] = manager.list()
                elif self.predict:
                    dictionary[i] = self.total_dict[i]
                    array_dict[i] = manager.list()
                else:
                    dictionary[i] = manager.dict()
                if not self.predict:
                    if isinstance(data[0][i], str):
                        p_list.append(Process(target=self.string_count, args=(data[:,i], self.total_dict[i], dictionary[i],)))
                    else:
                        p_list.append(Process(target=self.value_count, args=(data[:,i], self.total_dict[i], dictionary[i],)))
                else:
                    if isinstance(data[0][i], str):
                        p_list.append(Process(target=self.string_predict, args = (data[:, i], dictionary[i], array_dict[i],)))
                    else:
                        p_list.append(Process(target=self.value_predict, args = (data[:, i], dictionary[i], array_dict[i], i, )))
                
            for p in p_list:
                p.start()
            
            for p in p_list:
                p.join()
            
            if self.predict:
                for c in array_dict.keys():
                    for it in array_dict[c]:
                        prediction[c].append(str(it))
            else:
                for d in dictionary.keys():
                    save_obj(dict(dictionary[d]), 'counters/' + self.model_name + '_' + u + '/dict_' + str(d))

        for d in self.total_dict.keys():
            save_obj(dict(self.total_dict[d]), self.total_dict_file + '_' + str(d))

        if not self.predict:
            return None
        else:
            return prediction

    def string_count(self, col, t_d, d):
        for i in col:
            for n in i.split():
                if n in d.keys():
                    d[n] += 1
                else:
                    d[n] = 1
                if n in t_d.keys():
                    t_d[n] += 1
                else:
                    t_d[n] = 1

    def value_count(self, col, t_d, d):
        unique, counts = np.unique(col, return_counts = True)
        counts = counts/1000000
        for c, val in enumerate(unique):
            if val in d.keys():
                d[val] += counts[c]
            else:
                d[val] = counts[c]
            if val in t_d.keys():
                t_d[val] += counts[c]
            else:
                t_d[val] = counts[c]

    def value_predict(self, col, d, a, i):
        s = sum(d.values())
        append = a.append
        for n in range(len(col)):
            try:
                append(d[col[n]] / s)
            except:
                append(0)
            # try:
            #     a.append(1-np.sqrt(d[col[n]]/s))
            # except:
            #     a.append(0)

    def string_predict(self, col, d, a):
        s = sum(d.values())
        for i in range(len(col)):
            total = 0
            cnt = 0
            try:
                for n in col[i].split():
                    total += d[n]
                    cnt += 1
                a.append(1 - np.sqrt((total/cnt)/s))
            except:
                a.append(0)