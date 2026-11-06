import unittest
from streamlit.testing.v1 import AppTest

class AppPages(unittest.TestCase):
    def test_all_sidebar_pages_render(self):
        pages=['Dashboard','Dataset Explorer','Transactions','Budget Planner','Investment Portfolio','AI Financial Insights','Deep Learning Lab','AI Financial Assistant','Reports','Settings']
        app=AppTest.from_file('app.py',default_timeout=90).run()
        self.assertFalse(app.exception)
        for page in pages[1:]:
            app.radio[0].set_value(page).run()
            self.assertFalse(app.exception, f'{page} raised {app.exception}')
            if page=='Dataset Explorer':
                self.assertGreaterEqual(len(app.tabs),4)
                self.assertGreaterEqual(len(app.get('download_button')),4)

if __name__=='__main__': unittest.main()
