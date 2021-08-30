def normal_query(start_date, end_date, limit):
    sql = """select
                logtime, src_ip, dst_ip, max(end_time) as end,
                uniqExact(dst_port) as dst_port_cnt,

                avg(if(extract(toString(http_retcode), '10')=='10', 1, 0)) as info_st,
                avg(if(extract(toString(http_retcode), '20')=='20', 1, 0)) as succ_st,
                avg(if(extract(toString(http_retcode), '30')=='30', 1, 0)) as redir_st,
                avg(if(extract(toString(http_retcode), '40')=='40', 1, 0)) as cler_st,
                avg(if(extract(toString(http_retcode), '50')=='50', 1, 0)) as serer_st,
                avg(if(toString(http_retcode)=='', 1, 0)) as oth_st,

                -- arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_host, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as host_,
                arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_agent, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as agent_,
                arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_query, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as query_, 
                'normal' as label

            from dti.dti_qm_http_log
            where (logtime between toDateTime('2020-01-01 00:00:00') and toDateTime('2020-06-01 00:00:00'))
                    and not(http_host = '' and http_agent = ''  and http_query = '')
                    --and hash = 'normal'

            group by logtime, src_ip, dst_ip
            limit {limit}
            """.replace('{start_date}', start_date).replace('{end_date}', end_date).replace('{limit}', str(limit))
    return sql

def predict_query(start_date, end_date, limit):
    sql = """select
                logtime, src_ip, dst_ip, max(end_time) as end,
                uniqExact(dst_port) as dst_port_cnt,

                avg(if(extract(toString(http_retcode), '10')=='10', 1, 0)) as info_st,
                avg(if(extract(toString(http_retcode), '20')=='20', 1, 0)) as succ_st,
                avg(if(extract(toString(http_retcode), '30')=='30', 1, 0)) as redir_st,
                avg(if(extract(toString(http_retcode), '40')=='40', 1, 0)) as cler_st,
                avg(if(extract(toString(http_retcode), '50')=='50', 1, 0)) as serer_st,
                avg(if(toString(http_retcode)=='-', 1, 0)) as oth_st,

                -- arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_host, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as host_,
                arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_agent, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as agent_,
                arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_query, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as query_
            
            from dti.dti_qm_http_log
            --from dti.dti_sh_demo_log
            where (logtime between toDateTime('{start_date}') and toDateTime('{end_date}'))
                    and not(http_host = '' and http_agent = ''  and http_query = '')
                    --and hash != 'XSS'
                    --and hash != 'BEACONING'
                    --and hash != 'SQL_INJECTION'
                    --and hash != 'CREDENTIAL'
                    --and hash != 'normal'

            group by logtime, src_ip, dst_ip
            --limit {limit}

            """.format(start_date=start_date, end_date=end_date, limit = limit)
    return sql

def attack_split(y_):
    if y_.find(')') != -1:
        return y_[y_.find(')')+1:]
    else:
        return y_

def attack_sql_query(data_limit):
    main ="""select
                logtime, src_ip, dst_ip, max(end_time) as end,
                uniqExact(dst_port) as dst_port_cnt,

                avg(if(extract(toString(http_retcode), '10')=='10', 1, 0)) as info_st,
                avg(if(extract(toString(http_retcode), '20')=='20', 1, 0)) as succ_st,
                avg(if(extract(toString(http_retcode), '30')=='30', 1, 0)) as redir_st,
                avg(if(extract(toString(http_retcode), '40')=='40', 1, 0)) as cler_st,
                avg(if(extract(toString(http_retcode), '50')=='50', 1, 0)) as serer_st,
                avg(if(toString(http_retcode)=='', 1, 0)) as oth_st,

                -- arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_host, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as host_,
                arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_agent, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as agent_,
                arrayStringConcat(groupUniqArray(arrayStringConcat(extractAll(replaceRegexpAll(replaceAll(http_query, '.', ' ddott '), '[%/!@#$?,;:&*)(-+=]', ' '), '[a-zA-Z ]'))), ' ') as query_, 
                

            """

    beaconing = """
                     'file_download' as label
                from dti.dti_sh_demo_log
                where hash = 'BEACONING'
                    and not(http_host = '' and http_agent = '' and http_query = '')
                    and src_ip global not in
                            (
                                select distinct ip
                                from dti.dti_ip_ast
                            )
                    and src_ip like '%.%.%.%'
                group by logtime, src_ip, dst_ip
                limit {limit}

                """.replace('{limit}', str(data_limit))

    credential = """
                     'credential' as label
                from dti.dti_sh_demo_log
                where hash = 'CREDENTIAL'
                    and not(http_host = '' and http_agent = '' and http_query = '')
                    and src_ip global not in
                            (
                                select distinct ip
                                from dti.dti_ip_ast
                            )
                    and src_ip like '%.%.%.%'
                group by logtime, src_ip, dst_ip
                limit {limit}

                """.replace('{limit}', str(data_limit))


    xss = """
                    'xss' as label
                from dti.dti_sh_demo_log
                where hash = 'XSS'
                    and not(http_host = '' and http_agent = '' and http_query = '')
                    and src_ip global not in
                            (
                                select distinct ip
                                from dti.dti_ip_ast
                            )
                    and src_ip like '%.%.%.%'
                group by logtime, src_ip, dst_ip
                limit {limit}

                """.replace('{limit}', str(data_limit))

    injection = """
                        'injection' as label
                from dti.dti_sh_demo_log
                where hash = 'SQL_INJECTION'
                    and not(http_host = '' and http_agent = '' and http_query = '')
                    and src_ip global not in
                            (
                                select distinct ip
                                from dti.dti_ip_ast
                            )
                    and src_ip like '%.%.%.%'
                group by logtime, src_ip, dst_ip
                limit {limit}

                """.replace('{limit}', str(data_limit))

    return [main+beaconing, main+injection, main+credential, main+xss]