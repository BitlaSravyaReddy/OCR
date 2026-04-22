#define MyAppName "Invoice Batch Processor"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "XO Rig"
#define MyAppExeName "InvoiceBatchProcessor.exe"

[Setup]
AppId={{E05D3286-1A1B-45DF-A7C2-5B69A34E31E7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\InvoiceBatchProcessor
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\installer_output
OutputBaseFilename=InvoiceBatchProcessorSetup_Bundled
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked
Name: "startup"; Description: "Start Invoice Batch Processor when Windows starts"; GroupDescription: "Startup:"; Flags: checkedonce

[Files]
Source: "..\dist_installer\InvoiceBatchProcessor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\release_assets\paddle_models\*"; DestDir: "{app}\paddle_models"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "InvoiceBatchProcessor"; ValueData: """{app}\{#MyAppExeName}"" --start-minimized"; Flags: uninsdeletevalue; Tasks: startup

[InstallDelete]
Type: filesandordirs; Name: "{app}\*"
Type: files; Name: "{userstartup}\InvoiceBatchProcessor.cmd"

[UninstallDelete]
Type: files; Name: "{userstartup}\InvoiceBatchProcessor.cmd"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
