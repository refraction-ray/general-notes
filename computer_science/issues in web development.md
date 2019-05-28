# Web development: issues summary

*In this file, issues occured in web development and their solutions or workaournds are listed. Flask, sqlalchemy, vue are several important aspects in this note.*

*The main part is finished around 2018.12 to 2019.01, when I actively develop myarxiv.club.*

## Solved

* mysql encoding utf8 utf8mb4 3byte vs. 4byte. always use utf8mb4
* mysql encoding the change of tables' default encoding doesn't imply the changing encoding of existing column, need to change them too!
* relative import and script within module. python -m and outside the package to run the command `python -m mod.app`
* must run two celery daemon to implement the periodic tasks, namely indepedent celery beat.
* different max size of varchar in terms of encoding scheme in mysql.
* change charset from sqlalchemy side `__table_args__ = {'mysql_charset': "utf8mb4"}`
* sqlalchemy table args with both tuple and dict terms: [doc](https://docs.sqlalchemy.org/en/latest/orm/extensions/declarative/table_config.html#declarative-table-args), firstly tuple elements, the last tuple element is a dict
* for flask-sqlalchemy, if `__table_args__` is specified, the `__tablename__` attr should also be specified since it cannot be generated automatically somehow.
* `url_for` in flask must add the blueprint name as `bp.urlprefix`
* yaml: no tab for indent!!! syntax of yaml: https://juejin.im/post/5b2a3f32f265da598223d2bb
* jinja: comma separate list: [so](https://stackoverflow.com/questions/11974318/how-to-output-a-comma-delimited-list-in-jinja-python-template) `{{ "," if not loop.last }}`
* python path is refered from the script position, not the module position, try instead use `os.path.abspath`
* vue will automatically strip out all attrs in html label after rendering, but it still works! [so](https://stackoverflow.com/questions/48895041/vue-js-directives-being-stripped-out-of-html)
* python boolean in jinja is not js boolean in html! use a if condition transform it. [so](https://stackoverflow.com/questions/8433450/why-doesnt-my-condition-logic-work-as-expected-in-jinja2-cherrypy)
* flask must specify the content type to recoginize json data by `request.json`, which can only be set by `$.ajax` instead of the short cut `$.post` in js side.
* `after_app_request` for app range hooks though defined in blueprint [so](https://stackoverflow.com/questions/45368995/application-wide-request-hooks-in-flask-how-to-implement)
* development parameter in env variables
* `this` in done function of promise point to where? always use `var self=this` and use self in callback: js
* sqlalchemy: try: commit, except: rollback!! Never forget error may happen.
* pytest fixture for database: [doc](https://docs.sqlalchemy.org/en/latest/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites)
* celery apply task.run() run celery task locally
* Falsk: `request.json` can be None!!! while the `request.form` is always there as a ImmutableDict even its empty.
* `request.form.get` are all strings, no matter it is integer or boolean before, WTForm tranform the boolean value is default "false" to False, which can be set by false_values = ("false", "False"); On the contrary, the `request.json` object is consistent with python's data objects. Number is number and boolean is boolean for both in js and python. POST body for json method: `b'{"a": 1, "b": true}'`, POST body for data(form) method: `a=1&b=True`. GET param is the same as POST form. Therefore, the bool data send in form data form is different in js and python side. js side has a true boolean while python gives a True in the string. None is null when coded in json in python. List support in the data form. In server side, `request.form.getlist('name')`, client side, data body `a=1&a=2`, nested dict is not supported in data. `application/x-www-form-urlencoded` vs `applocation/json`
* During tests, sqlalchemy.exc.ResourceClosedError: This Connection is closed, the db is reused across different functions, so you need add it on each function, otherwise it is already closed in the last function
* the smtp handler in logging module doesn't support ssl directly, one should hack it with subclass [so](https://stackoverflow.com/questions/36937461/how-can-i-send-an-email-using-python-loggings-smtphandler-and-ssl).
* the exact behavior of the import system in python…, see my [blog](https://re-ra.xyz/Python-%E7%9A%84-import-%E7%A9%B6%E7%AB%9F%E5%81%9A%E4%BA%86%E4%BB%80%E4%B9%88/)
* the config name in celery are some beginning with celery_ , while some others not, refer on this table in [doc](http://docs.celeryproject.org/en/latest/userguide/configuration.html#new-lowercase-settings), be careful
* 554 DT:SPM 163 mail, 163 easily take your email as spoof, no good solution if you are in bad luck
* flask admin: MixIN must before the view class to override is_accessible
* sqlalchemy `casacades="delet orphans" ` for 1 to many relations
* celery worker running in pytest, by comment one line in celery, which makes nosense check. If you want to leave it alone, then use some function in test py file hacking it



## Partially solved

* circular reference between celery and app



## Unsolved

* bulk insert by sqlalchemy side without breaking model level relations (not a good idea itself, if there is the possibility of failure which should always be considered.)

* insert ignore by sqlalchemy without breaking model level relations (basically no hope)

  > SQLAlchemy does not provide an interface to `ON DUPLICATE KEY UPDATE` or `MERGE` or any other similar functionality in its ORM layer. Nevertheless, it has the [`session.merge()`](http://docs.sqlalchemy.org/en/rel_0_7/orm/session.html#merging) function that can replicate the functionality **only if the key in question is a primary key**. [so](https://stackoverflow.com/questions/6611563/sqlalchemy-on-duplicate-key-update)