import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from sphinx.ext.autodoc import ModuleLevelDocumenter, DataDocumenter

def add_directive_header(self, sig):
  ModuleLevelDocumenter.add_directive_header(self, sig)
  # Rest of original method ignored

DataDocumenter.add_directive_header = add_directive_header

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.githubpages']
autodoc_mock_imports = ['numpy', 'scipy', 'tensorflow',
                        'python_speech_features', 'soundfile', 'editdistance']
autodoc_member_order = 'bysource'
napoleon_include_init_with_doc = True

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = 'Korean Speech Recognition Using Deep Learning'
copyright = '2018, Suji Lee and Seokjin Han'
author = 'Suji Lee and Seokjin Han'

version = ''
release = ''
language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False

html_theme = 'alabaster'
# html_theme_options = {}
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    # 'preamble': '',
    # 'figure_align': 'htbp',
}
latex_documents = [
    (master_doc, 'manual.tex',
     'Korean Speech Recognition Using Deep Learning',
     'Suji Lee and Seokjin Han', 'howto'),
]
