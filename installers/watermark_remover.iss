[Setup]
AppName=Watermark Remover Suite
AppVersion=0.1.0
DefaultDirName={pf}\WatermarkRemoverSuite
DefaultGroupName=Watermark Remover Suite
OutputDir=installers\build
OutputBaseFilename=WatermarkRemoverSuite_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "..\dist\WatermarkRemoverSuite\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Watermark Remover CLI"; Filename: "{app}\WatermarkRemoverSuite.exe"
Name: "{group}\Uninstall Watermark Remover Suite"; Filename: "{uninstallexe}"
