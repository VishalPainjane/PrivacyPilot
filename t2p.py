import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def txt_to_pdf(txt_file, pdf_file):
    try:
        with open(txt_file, 'r', encoding='utf-8') as file:
            text_content = file.read()
        
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()
        
        paragraphs = []
        for line in text_content.split('\n'):
            if line.strip():
                p = Paragraph(line, styles['Normal'])
                paragraphs.append(p)
                paragraphs.append(Spacer(1, 6))
        
        doc.build(paragraphs)
        print(f"Successfully converted {txt_file} to {pdf_file}")
        return True
    
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) == 3:
        txt_to_pdf(sys.argv[1], sys.argv[2])
    else:
        txt_file = 't1.txt'
        pdf_file = 'p1.pdf'

        txt_to_pdf(txt_file, pdf_file)
