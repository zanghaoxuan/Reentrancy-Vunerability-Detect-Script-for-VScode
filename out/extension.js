"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
const vscode = __importStar(require("vscode"));
const child_process_1 = require("child_process");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
function activate(context) {
    console.log("[DEBUG] Script is activated");
    const command = vscode.commands.registerCommand('extension.detectReentrancy', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor)
            return;
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
        const py = (0, child_process_1.spawn)(pythonPath, ['VoteRD_predict.py', path.join('predict_input', 'tmp.sol')], {
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
            if (text.includes("No reentrancy issues found")) {
                console.log("[INFO] DONE!");
                finalResult = "✅ No reentrancy issues found!";
                py.kill();
            }
        });
        py.on('close', (code) => {
            if (finalResult) {
                vscode.window.showInformationMessage(finalResult);
            }
            else {
                vscode.window.showWarningMessage("❓ Please cheak the log, maybe something wrong");
            }
        });
    });
    context.subscriptions.push(command);
}
//# sourceMappingURL=extension.js.map