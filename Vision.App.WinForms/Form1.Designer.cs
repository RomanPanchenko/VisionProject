namespace Vision.App.WinForms;

partial class Form1
{
    /// <summary>
    ///  Required designer variable.
    /// </summary>
    private System.ComponentModel.IContainer components = null;

    /// <summary>
    ///  Clean up any resources being used.
    /// </summary>
    /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing && (components != null))
        {
            components.Dispose();
        }
        base.Dispose(disposing);
    }

    #region Windows Form Designer generated code

    /// <summary>
    ///  Required method for Designer support - do not modify
    ///  the contents of this method with the code editor.
    /// </summary>
    private void InitializeComponent()
    {
        this.components = new System.ComponentModel.Container();
        tableLayoutPanel1 = new TableLayoutPanel();
        drawingCanvas1 = new DrawingCanvas();
        rightPanel = new Panel();
        lblFeedbackHint = new Label();
        flowFeedbackButtons = new FlowLayoutPanel();
        lblFeedbackStatus = new Label();
        btnClear = new Button();
        btnLoadModel = new Button();
        lblPred = new Label();
        lblTop = new Label();
        lblModel = new Label();
        picturePreview = new PictureBox();
        tableLayoutPanel1.SuspendLayout();
        rightPanel.SuspendLayout();
        ((System.ComponentModel.ISupportInitialize)picturePreview).BeginInit();
        SuspendLayout();
        // 
        // tableLayoutPanel1
        // 
        tableLayoutPanel1.ColumnCount = 2;
        tableLayoutPanel1.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 70F));
        tableLayoutPanel1.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 30F));
        tableLayoutPanel1.Controls.Add(drawingCanvas1, 0, 0);
        tableLayoutPanel1.Controls.Add(rightPanel, 1, 0);
        tableLayoutPanel1.Dock = DockStyle.Fill;
        tableLayoutPanel1.Location = new Point(0, 0);
        tableLayoutPanel1.Name = "tableLayoutPanel1";
        tableLayoutPanel1.RowCount = 1;
        tableLayoutPanel1.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
        tableLayoutPanel1.Size = new Size(1100, 760);
        tableLayoutPanel1.TabIndex = 0;
        // 
        // drawingCanvas1
        // 
        drawingCanvas1.Dock = DockStyle.Fill;
        drawingCanvas1.Location = new Point(8, 8);
        drawingCanvas1.Margin = new Padding(8);
        drawingCanvas1.Name = "drawingCanvas1";
        drawingCanvas1.Size = new Size(754, 744);
        drawingCanvas1.TabIndex = 0;
        // 
        // rightPanel
        // 
        rightPanel.Controls.Add(picturePreview);
        rightPanel.Controls.Add(lblFeedbackStatus);
        rightPanel.Controls.Add(flowFeedbackButtons);
        rightPanel.Controls.Add(lblFeedbackHint);
        rightPanel.Controls.Add(lblModel);
        rightPanel.Controls.Add(lblTop);
        rightPanel.Controls.Add(lblPred);
        rightPanel.Controls.Add(btnLoadModel);
        rightPanel.Controls.Add(btnClear);
        rightPanel.Dock = DockStyle.Fill;
        rightPanel.Location = new Point(694, 8);
        rightPanel.Margin = new Padding(8);
        rightPanel.Name = "rightPanel";
        rightPanel.Size = new Size(322, 744);
        rightPanel.TabIndex = 1;

        // 
        // lblFeedbackHint
        // 
        lblFeedbackHint.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        lblFeedbackHint.Location = new Point(8, 330);
        lblFeedbackHint.Name = "lblFeedbackHint";
        lblFeedbackHint.Size = new Size(262, 36);
        lblFeedbackHint.TabIndex = 5;
        lblFeedbackHint.Text = "Если сеть ошиблась — нажмите правильный класс (цифра или буква):";

        // 
        // flowFeedbackButtons
        // 
        flowFeedbackButtons.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Bottom;
        flowFeedbackButtons.AutoScroll = true;
        flowFeedbackButtons.Location = new Point(8, 370);
        flowFeedbackButtons.Name = "flowFeedbackButtons";
        flowFeedbackButtons.Size = new Size(306, 260);
        flowFeedbackButtons.TabIndex = 6;

        // 
        // lblFeedbackStatus
        // 
        lblFeedbackStatus.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        lblFeedbackStatus.Location = new Point(8, 696);
        lblFeedbackStatus.Name = "lblFeedbackStatus";
        lblFeedbackStatus.Size = new Size(262, 48);
        lblFeedbackStatus.TabIndex = 7;
        lblFeedbackStatus.Text = "";
        // 
        // btnClear
        // 
        btnClear.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        btnClear.Location = new Point(8, 8);
        btnClear.Name = "btnClear";
        btnClear.Size = new Size(262, 34);
        btnClear.TabIndex = 0;
        btnClear.Text = "Очистить";
        btnClear.UseVisualStyleBackColor = true;
        // 
        // btnLoadModel
        // 
        btnLoadModel.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        btnLoadModel.Location = new Point(8, 50);
        btnLoadModel.Name = "btnLoadModel";
        btnLoadModel.Size = new Size(262, 34);
        btnLoadModel.TabIndex = 1;
        btnLoadModel.Text = "Загрузить модель...";
        btnLoadModel.UseVisualStyleBackColor = true;
        // 
        // lblPred
        // 
        lblPred.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        lblPred.Font = new Font("Segoe UI", 18F, FontStyle.Bold);
        lblPred.Location = new Point(8, 100);
        lblPred.Name = "lblPred";
        lblPred.Size = new Size(262, 42);
        lblPred.TabIndex = 2;
        lblPred.Text = "Предсказание: -";
        // 
        // lblTop
        // 
        lblTop.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        lblTop.Location = new Point(8, 150);
        lblTop.Name = "lblTop";
        lblTop.Size = new Size(262, 110);
        lblTop.TabIndex = 3;
        lblTop.Text = "top-3: -";
        // 
        // lblModel
        // 
        lblModel.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        lblModel.Location = new Point(8, 270);
        lblModel.Name = "lblModel";
        lblModel.Size = new Size(262, 56);
        lblModel.TabIndex = 4;
        lblModel.Text = "Модель: (не загружена)";
        // 
        // picturePreview
        // 
        picturePreview.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        picturePreview.Location = new Point(8, 640);
        picturePreview.Name = "picturePreview";
        picturePreview.Size = new Size(306, 48);
        picturePreview.SizeMode = PictureBoxSizeMode.Zoom;
        picturePreview.TabIndex = 8;
        picturePreview.TabStop = false;
        // 
        // Form1
        // 
        AutoScaleMode = AutoScaleMode.Font;
        ClientSize = new Size(1100, 760);
        Controls.Add(tableLayoutPanel1);
        MinimumSize = new Size(820, 560);
        Name = "Form1";
        Text = "Vision MNIST Draw";
        tableLayoutPanel1.ResumeLayout(false);
        rightPanel.ResumeLayout(false);
        ((System.ComponentModel.ISupportInitialize)picturePreview).EndInit();
        ResumeLayout(false);
    }

    #endregion

    private TableLayoutPanel tableLayoutPanel1;
    private DrawingCanvas drawingCanvas1;
    private Panel rightPanel;
    private Button btnClear;
    private Button btnLoadModel;
    private Label lblPred;
    private Label lblTop;
    private Label lblModel;
    private Label lblFeedbackHint;
    private FlowLayoutPanel flowFeedbackButtons;
    private Label lblFeedbackStatus;
    private PictureBox picturePreview;
}
