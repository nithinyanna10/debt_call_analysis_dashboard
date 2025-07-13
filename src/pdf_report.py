from fpdf import FPDF

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "📞 Call Prediction Report", ln=True, align="C")

    def add_table(self, df):
        self.set_font("Arial", size=10)
        col_widths = [30] * len(df.columns)

        for col in df.columns:
            self.cell(col_widths[0], 8, col[:12], border=1)
        self.ln()

        for _, row in df.iterrows():
            for val in row:
                self.cell(col_widths[0], 8, str(val)[:15], border=1)
            self.ln()

def generate_pdf_report(df, path="report.pdf"):
    pdf = ReportPDF()
    pdf.add_page()
    pdf.add_table(df)
    pdf.output(path)
    return path
