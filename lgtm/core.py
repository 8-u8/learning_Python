import click


@click.command()
@click.option('--message', '-m', default='LGTM',
              show_default=True, help='画像に乗せる文字列')
def cli(keyword, message):
    '''
    LGTM自動生成ツール
    '''
    lgtm(keyword, message)
    click.echo('lgtm')


def lgtm():
    pass
