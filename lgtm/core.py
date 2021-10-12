import click


@click.command()
def cli():
    '''
    LGTM自動生成ツール
    '''
    lgtm()
    click.echo('lgtm')


def lgtm():
    pass
