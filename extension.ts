import * as vscode from 'vscode';
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

export function activate(context: vscode.ExtensionContext) {
    console.log("[DEBUG] Script is activated");
    const command = vscode.commands.registerCommand('extension.detectReentrancy', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selection = editor.document.getText(editor.selection);
        const inputDir = path.join(__dirname, '..', 'predict_input');
        const inputFilePath = path.join(inputDir, 'tmp.sol');

        // 确保目录存在
        if (!fs.existsSync(inputDir)) {
            fs.mkdirSync(inputDir);
        }

        console.log("[DEBUG] Writing Input File...");
        fs.writeFileSync(inputFilePath, selection, 'utf-8');
        console.log("[DEBUG] Input File Written");

        const pythonPath = process.platform === 'win32'
            ? path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe')
            : path.join(__dirname, '..', 'venv', 'bin', 'python');

        
        console.log("[DEBUG] Loadign VoteRD...");
        const py = spawn(pythonPath, ['VoteRD_predict.py', path.join('predict_input', 'tmp.sol')], {
            cwd: path.join(__dirname, '..')
        });
        console.log("[DEBUG] VoteRD loaded");
        console.log("[INFO] Wait For A While...");
        
        let output = '';
        let finalResult = '';
        py.on('error', (err) => {
            console.error(`[Python ERROR] Failed to start process: ${err.message}`);
        });
        py.stdout.on('data', (data) => {
            const text = data.toString();
            output += text;
            console.log(`[Python stdout] ${text.trim()}`);

            if (text.includes("Reentrancy vulnerability detected")) {
                
                console.log("[INFO] DONE!");
                finalResult = "⚠️ Reentrancy vulnerability detected!";
                py.kill();
            }

            if(text.includes("No reentrancy issues found")){
                console.log("[INFO] DONE!");
                finalResult = "✅ No reentrancy issues found!";
                py.kill();
            
            }
        });

        py.on('close', (code) => {
            if (finalResult) {
                vscode.window.showInformationMessage(finalResult);
            } else {
                vscode.window.showWarningMessage("❓ Please cheak the log, maybe something wrong");
            }
        });
    });

    context.subscriptions.push(command);
}
