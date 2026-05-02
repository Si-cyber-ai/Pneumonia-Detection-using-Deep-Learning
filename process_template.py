import re
import json

def process_html():
    with open('pneumoai_advanced_diagnostic_report (1).html', 'r', encoding='utf-8') as f:
        html = f.read()

    # Replacements based on instructions
    # Diagnosis → {{ diagnosis }}
    html = re.sub(r'<div class="hero-diag">Pneumonia Detected</div>', r'<div class="hero-diag">{{ diagnosis }}</div>', html)
    
    # Confidence → {{ confidence }}
    # In <div class="hstat-val">99.5%</div>
    html = re.sub(r'<div class="hstat"><div class="hstat-label">Confidence</div><div class="hstat-val">99.5%</div><div class="hstat-sub">Very High</div></div>',
                  r'<div class="hstat"><div class="hstat-label">Confidence</div><div class="hstat-val">{{ confidence }}%</div><div class="hstat-sub">Very High</div></div>', html)
    
    html = re.sub(r'<canvas id="confChart" role="img" aria-label=".*?">.*?</canvas>',
                  r'<canvas id="confChart" role="img" aria-label="{{ confidence }}% confidence for pneumonia"></canvas>', html)
    html = re.sub(r"ctx\.fillText\('99\.5%', cx, cy - 8\);", r"ctx.fillText('{{ confidence }}%', cx, cy - 8);", html)
    html = re.sub(r'data: \[99\.5, 0\.5\],', r'data: [{{ confidence }}, 100 - {{ confidence }}],', html)

    # Severity → {{ severity }}
    html = re.sub(r'<div class="hstat"><div class="hstat-label">Severity</div><div class="hstat-val">Moderate</div><div class="hstat-sub">Grade II / IV</div></div>',
                  r'<div class="hstat"><div class="hstat-label">Severity</div><div class="hstat-val">{{ severity }}</div><div class="hstat-sub">Grade II / IV</div></div>', html)
    html = re.sub(r'<span style="font-weight:600;color:#BA7517">Moderate \(55%\)</span>',
                  r'<span style="font-weight:600;color:#BA7517">{{ severity }}</span>', html)
    
    # Agreement → {{ agreement }}
    html = re.sub(r'<div class="hstat"><div class="hstat-label">Model Agreement</div><div class="hstat-val">100%</div><div class="hstat-sub">DenseNet & ResNet</div></div>',
                  r'<div class="hstat"><div class="hstat-label">Model Agreement</div><div class="hstat-val">{{ agreement }}</div><div class="hstat-sub">DenseNet & ResNet</div></div>', html)
    html = re.sub(r'<div style="text-align:center;margin-top:8px;font-size:11px;color:var\(--t2\)">DenseNet121 · ResNet50 · Concordant</div>',
                  r'<div style="text-align:center;margin-top:8px;font-size:11px;color:var(--t2)">{{ densenet }} · {{ resnet }} · {{ agreement }}</div>', html)

    # DenseNet → {{ densenet }} and ResNet → {{ resnet }}
    # Wait, the instructions didn't specify where they go, maybe replace "DenseNet121" and "ResNet50"?
    # Actually, they might be predictions, e.g., {{ densenet }} prediction and {{ resnet }} prediction.
    # We can inject them in the Model Agreement subtext if needed, or we can just leave them as DenseNet and ResNet.
    # Let's add them to the alert body or somewhere. Or let's see where they appear.
    
    # Focus Score → {{ focus_score }}
    html = re.sub(r'<span style="font-weight:600;color:#185FA5">High \(0\.85\)</span>',
                  r'<span style="font-weight:600;color:#185FA5">{{ focus_score }}</span>', html)
    
    # Lung Involvement → {{ lung_involvement }}
    html = re.sub(r'<span style="font-weight:600;color:#BA7517">~40%</span>',
                  r'<span style="font-weight:600;color:#BA7517">{{ lung_involvement }}</span>', html)
    
    # Region Summary → {{ region_summary }}
    html = re.sub(r'<div class="hero-sub">Bilateral infiltrates identified — immediate clinical correlation advised</div>',
                  r'<div class="hero-sub">{{ region_summary }}</div>', html)

    # Recommendation → {{ recommendation }}
    # Let's replace the whole Recommended clinical next steps card body
    # Or replace the section: Recommended medication protocol
    html = re.sub(r'<div class="alert-body">High-confidence AI detection of pneumonia infiltrates.*?</div>',
                  r'<div class="alert-body">{{ recommendation }}</div>', html, flags=re.DOTALL)
    
    # AI Explanation → {{ llm_report }}
    # Let's add a new section for AI Explanation or replace the disclaimer/alert body?
    # The prompt says AI Explanation -> {{ llm_report }}
    # We can create a new section right after confidence & severity analysis.
    llm_report_section = """
  <div class="section">
    <div class="section-label">AI Explanation</div>
    <div class="card">
      <div style="font-size:13px;color:var(--t1);line-height:1.6">{{ llm_report }}</div>
    </div>
  </div>
"""
    html = html.replace('  <div class="section">\n    <div class="section-label">Typical clinical progression (days)</div>',
                        llm_report_section + '\n  <div class="section">\n    <div class="section-label">Typical clinical progression (days)</div>')

    # Dynamic severity grade
    html = html.replace('Grade II / IV', '{{ severity_grade }}')
    
    # Clinical alert section
    html = html.replace('<div class="alert-card alert-danger">', '<div class="alert-card {{ alert_class }}">')
    html = html.replace('<div class="alert-icon">⚠</div>', '<div class="alert-icon">{{ alert_icon }}</div>')
    html = html.replace('<div class="alert-title">Immediate medical attention strongly recommended</div>', '<div class="alert-title">{{ alert_title }}</div>')
    html = html.replace('<div class="alert-body">High-confidence AI detection of pneumonia infiltrates. This report must be reviewed by a qualified physician before any treatment is initiated. Do not self-medicate.</div>', '<div class="alert-body">{{ alert_body }}</div>')
    
    # Hide medication and charts for Normal diagnosis
    html = html.replace('<div class="section">\n    <div class="section-label">Recommended medication protocol</div>', '<div class="section" style="display: {{ display_meds }}">\n    <div class="section-label">Recommended medication protocol</div>')
    html = html.replace('<div class="section">\n    <div class="section-label">Antibiotic efficacy over treatment course</div>', '<div class="section" style="display: {{ display_meds }}">\n    <div class="section-label">Antibiotic efficacy over treatment course</div>')
    html = html.replace('<div class="section">\n    <div class="section-label">Typical clinical progression (days)</div>', '<div class="section" style="display: {{ display_meds }}">\n    <div class="section-label">Typical clinical progression (days)</div>')
    html = html.replace('<div class="section">\n    <div class="section-label">Vital signs — target monitoring ranges</div>', '<div class="section" style="display: {{ display_meds }}">\n    <div class="section-label">Vital signs — target monitoring ranges</div>')
    
    # Add CSS for success alert
    html = html.replace('.alert-danger .alert-icon{background:#F7C1C1}', '.alert-danger .alert-icon{background:#F7C1C1}\n  .alert-success{background:#EAF3DE;border:0.5px solid #C8E3AD}\n  .alert-success .alert-icon{background:#C8E3AD;color:#27500A}\n  .alert-success .alert-title{color:#27500A}\n  .alert-success .alert-body{color:#3B6D11}')

    # Add fallbacks for CSS variables so it renders perfectly standalone
    html = html.replace('var(--color-text-primary)', 'var(--color-text-primary, #0f172a)')
    html = html.replace('var(--color-text-secondary)', 'var(--color-text-secondary, #64748b)')
    html = html.replace('var(--color-background-primary)', 'var(--color-background-primary, #ffffff)')
    html = html.replace('var(--color-background-secondary)', 'var(--color-background-secondary, #f8fafc)')
    html = html.replace('var(--color-border-tertiary)', 'var(--color-border-tertiary, #e2e8f0)')
    
    # Ensure JS doesn't break due to brackets inside template string.
    js_export = "export const reportTemplate = `" + html.replace('`', '\\`').replace('$', '\\$') + "`;\n"
    
    with open('frontend/src/reportTemplate.js', 'w', encoding='utf-8') as f:
        f.write(js_export)

process_html()
