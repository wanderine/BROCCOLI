#!/usr/bin/python

# Handle unicode encoding

import locale
locale.setlocale(locale.LC_ALL, '')
code = locale.getpreferredencoding()

import curses

from optparse import OptionParser, OptionGroup, SUPPRESS_HELP

fsli_C_FAILED = 1
fsli_C_OK = 2
fsli_C_SKIP = 4
fsli_C_WARN = 3

class Version(object):
    def __init__(self,version_string):
        v_vals = version_string.split('.')
        for v in v_vals:
            if not v.isdigit():
                raise ValueError('Bad version string')
        self.major = int(v_vals[0])
        try:
            self.minor = int(v_vals[1])
        except IndexError:
            self.minor = 0
        try:
            self.patch = int(v_vals[2])
        except IndexError:
            self.patch = 0
        try:
            self.hotfix = int(v_vals[3])
        except IndexError:
            self.hotfix = 0
        
    def __repr__(self):
        return "Version(%s,%s,%s,%s)" % (self.major, self.minor, self.patch, self.hotfix)
    def __str__(self):
        if self.hotfix == 0:
            return "%s.%s.%s" % (self.major, self.minor, self.patch)
        else:
            return "%s.%s.%s.%s" % (self.major, self.minor, self.patch, self.hotfix)
    def __ge__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self > other or self == other:
            return True
        return False
    def __le__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self < other or self == other:
            return True
        return False
    def __cmp__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.__lt__(other):
            return -1
        if self.__gt__(other):
            return 1
        return 0
    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major < other.major:
            return True
        if self.major > other.major:
            return False
        if self.minor < other.minor:
            return True
        if self.minor > other.minor:
            return False
        if self.patch < other.patch:
            return True
        if self.patch > other.patch:
            return False
        if self.hotfix < other.hotfix:
            return True
        if self.hotfix > other.hotfix:
            return False
        # major, minor and patch all match so this is not less than
        return False
    
    def __gt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major > other.major:
            return True
        if self.major < other.major:
            return False
        if self.minor > other.minor:
            return True
        if self.minor < other.minor:
            return False
        if self.patch > other.patch:
            return True
        if self.patch < other.patch:
            return False
        if self.hotfix > other.hotfix:
            return True
        if self.hotfix < other.hotfix:
            return False
        # major, minor and patch all match so this is not less than
        return False 
    
    def __eq__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major == other.major and self.minor == other.minor and self.patch == other.patch and self.hotfix == other.hotfix:
            return True
        return False
    
    def __ne__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.__eq__(other):
            return False
        return True

version = Version('2.0.20')

class FslIResult(object):
    SUCCESS = 0
    WARN = 1
    ERROR = 2
    def __init__(self,result,status,message):
        self.result = result
        self.status = status
        self.message = message
    def __nonzero__(self):
        self.status


class Vendor(object):
    def __init__(self, name):
        self.name = name
        self.releases = []
        
class Platform(object):
    def __init__(self, name):
        self.name = name
        self.vendors={}
    def __repr__(self):
        self.name
    def addVendor(self,vname):
        self.vendors[vname] = Vendor(vname)

class InstallFailed(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)                     

def centred_x(scr, text):
    (_, width) = scr.getmaxyx()
    return width/2 - len(text)/2

def init_colours():
    curses.init_pair(fsli_C_FAILED, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(fsli_C_OK, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(fsli_C_WARN, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(fsli_C_SKIP, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

class shell_colours(object):
    default = '\033[0m'
    rfg_kbg = '\033[91m'
    gfg_kbg = '\033[92m'
    yfg_kbg = '\033[93m'
    mfg_kbg = '\033[95m'
    yfg_bbg = '\033[104;93m'
    bfg_kbg = '\033[34m'
    bold = '\033[1m'

class MsgUser(object): 
    __debug = False
    __quiet = False
    
    @classmethod
    def debugOn(cls):
        cls.__debug = True
    @classmethod
    def debugOff(cls):
        cls.__debug = False
    @classmethod
    def quietOn(cls):
        cls.__quiet = True
    @classmethod
    def quietOff(cls):
        cls.__quiet = False

    @classmethod
    def isquiet(cls):
        return cls.__quiet
    
    @classmethod
    def isdebug(cls):
        return cls.__debug
    
    @classmethod    
    def debug(cls, message, newline=True):
        if cls.__debug:
            from sys import stderr
            mess = str(message)
            if newline:
                mess += "\n"
            stderr.write(mess)
    
    @classmethod
    def message(cls, msg, tui=False):
        if cls.__quiet:
            return
        if tui:
            tui.info.addstr(msg)
            tui.info.refresh()
        else:
            print msg
    
    @classmethod
    def question(cls, msg, tui=False):
        if tui:
            pass
        else:
            print msg,
                  
    @classmethod
    def skipped(cls, msg, tui=False):
        if cls.__quiet:
            return
        if tui:
            tui.info.addstr("[Skipped] ", curses.A_BOLD | curses.color_pair(fsli_C_SKIP))
            tui.info.addstr(msg)
            tui.info.refresh()
        else:
            print "".join( (shell_colours.mfg_kbg, "[Skipped] ", shell_colours.default, msg ) )

    @classmethod
    def ok(cls, msg, tui=False):
        if cls.__quiet:
            return
        if tui:
            tui.info.addstr("[OK] ", curses.A_BOLD | curses.color_pair(fsli_C_OK))
            tui.info.addstr(msg)
            tui.info.refresh()
        else:
            print "".join( (shell_colours.gfg_kbg, "[OK] ", shell_colours.default, msg ) )
    
    @classmethod
    def failed(cls, msg, tui=False):
        if tui:
            tui.info.addstr("[FAILED] ", curses.A_BOLD | curses.color_pair(fsli_C_FAILED))
            tui.info.addstr(msg)
            tui.info.refresh()
        else:
            print "".join( (shell_colours.rfg_kbg, "[FAILED] ", shell_colours.default, msg ) )
    
    @classmethod
    def warning(cls, msg, tui=False):
        if cls.__quiet:
            return
        if tui:
            tui.info.addstr("[Warning] ", curses.A_BOLD | curses.color_pair(fsli_C_WARN) )
            tui.info.addstr(msg)
            tui.info.refresh()
        else:
            print "".join( (shell_colours.bfg_kbg, shell_colours.bold, "[Warning]", shell_colours.default, " ", msg ) )

class Progress_bar(object):
    def __init__(self, tui=False, x=0, y=0, mx=1, numeric=False):
        if tui:
            self.screen = tui.progress
        else:
            self.screen = False
        self.x = x
        self.y = y
        if self.screen:
            (_,self.width) = self.screen.getmaxyx()
        else:
            self.width = 50
        self.current = 0
        self.max = mx
        self.numeric = numeric
    def update(self, reading):
        from sys import stdout
        if MsgUser.isquiet():
            return
        percent = reading * 100 / self.max
        cr = '\r'
        if not self.screen:
            if not self.numeric:
                bar = '#' * int(percent)
            else:
                bar = "/".join((str(reading), str(self.max))) + ' - ' + str(percent) + "%\033[K"
            stdout.write(cr)
            stdout.write(bar)
            stdout.flush()
            self.current = percent
        else:
            bar = '#' * self.width
            nhash = int(self.width * (percent/100.0))
            self.screen.addnstr(self.y,self.x,bar,nhash)
        if percent == 100:
            stdout.write(cr)
            if not self.numeric:
                stdout.write(" " * int(percent))
                stdout.write(cr)
                stdout.flush()
            else:
                stdout.write(" " * ( len(str(self.max))*2 + 8))
                stdout.write(cr)
                stdout.flush()

def tempFileName(mode='r',close=False):
    '''Return a name for a temporary file - uses mkstemp to create the file and returns a tuple (file object, file name).
    Opens as read-only unless mode specifies otherwise. If close is set to True will close the file before returning.
    The file object is a fdopen file object so lacks a useable file name.'''
    from tempfile import mkstemp
    from os import fdopen
    
    (tmpfile, fname) = mkstemp()
    file_obj = fdopen(tmpfile, mode)

    if close:
        file_obj.close()
    return (file_obj, fname)

def run_cmd_countlines(command, maxnumber, tui=False, as_root=False ):
    '''Run the command and return result. Prints a progress bar scaled such that the number of output lines is a percentage of maxnumber'''
    import subprocess as sp
    
    command_line = command.split(' ')
    
    if as_root:
        command_line.insert(0, 'sudo')
    cmd = sp.Popen(command_line, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=0)

    prog = Progress_bar(mx = maxnumber, tui = tui)
    lines = 0
    while cmd.poll() is None:
        cmd.stdout.flush()
        _ = cmd.stdout.readline()
        lines += 1
        prog.update(lines)
    (output, error) = cmd.communicate()
    if cmd.returncode:
        return FslIResult('', FslIResult.ERROR, error)
    return FslIResult(output, FslIResult.SUCCESS, '')

def run_cmd_dropstdout(command, as_root=False):
    '''Run the command and return result.'''
    import subprocess as sp
    
    command_line = command.split(' ')
    
    if as_root:
        command_line.insert(0, 'sudo')
    cmd = sp.Popen(command_line, stdout=None, stderr=sp.PIPE)

    (_, error) = cmd.communicate()
    if cmd.returncode:
        return FslIResult('', FslIResult.ERROR, error)
    return FslIResult('', FslIResult.SUCCESS, '')

def run_cmd(command, as_root=False):
    '''Run the command and return result.'''
    import subprocess as sp
    
    command_line = command.split(' ')
    
    if as_root:
        command_line.insert(0, 'sudo')
    cmd = sp.Popen(command_line, stdout=sp.PIPE, stderr=sp.PIPE)
    MsgUser.debug("Will call %s" % (command_line))
    (output, error) = cmd.communicate()
    if cmd.returncode:
        MsgUser.debug("An error occured (%s, %s)" % (cmd.returncode, error))
        return FslIResult('', FslIResult.ERROR, error)
    MsgUser.debug("Command completed successfully (%s)" % (output))
    return FslIResult(output, FslIResult.SUCCESS, '')

def safe_delete(fs_object, as_root=False):
    '''Delete file/folder, becoming root if necessary. Run some sanity checks on object'''
    from os import path
    
    banned_items = [ '/', '/usr', '/usr/bin', '/usr/local', '/bin',
                     '/sbin', '/opt', '/Library', '/System', '/System/Library',
                     '/var', '/tmp', '/var/tmp', '/lib', '/lib64', '/Users',
                     '/home', '/Applications', '/private', '/etc', '/dev',
                     '/Network', '/net', '/proc']
    if path.isdir(fs_object):
        del_opts = "-rf"
    else:
        del_opts = '-f'

    if object in banned_items:
        return FslIResult('', FslIResult, 'Will not delete %s!' % (object))

    command_line = " ".join(('rm', del_opts, fs_object))
    return run_cmd(command_line, as_root)

def copy_file(fname, destination, as_root):
    '''Copy a file using sudo if necessary'''
    from os import path
    MsgUser.debug("Copying %s to %s (as root? %s)" %(fname, destination, as_root))
    if path.isdir(fname):
        return FslIResult('', FslIResult.ERROR, 'Source (%s) is not a file!' % (fname))

    if path.isdir(destination):
        # Ensure that copying into a folder we have a terminating slash
        destination = destination.rstrip('/') + "/"
    copy_opts = '-p'
    command_line = " ".join(('cp', copy_opts, fname, destination))
    
    return run_cmd(command_line, as_root)
    
def file_contains(fname, search_for):
    '''Equivalent of grep'''
    from re import compile,escape
    regex = compile(escape(search_for))
    found = False

    f = open(fname, 'r')
    for l in f:
        if regex.search(l):
            found = True
            break
    f.close()
    
    return found

def file_contains_1stline(search_for, fname):
    '''Equivalent of grep - returns first occurrence'''
    from re import compile, escape
    regex = compile(escape(search_for))
    found = ''
    MsgUser.debug("In file_contains_1stline.")
    MsgUser.debug("Looking for %s in %s." % (search_for, fname ))
    f = open(fname, 'r')
    for l in f:
        if regex.search(l):
            found = l
            break
    f.close()
    
    return found

def line_string_replace(line, search_for, replace_with):
    from re import sub,escape
    return sub(escape(search_for), escape(replace_with), line)
    
def line_starts_replace(line, search_for, replace_with):
    if line.startswith(search_for):
        return replace_with + '\n'
    return line

def move_file(from_file, to_file, requires_root = False):
    '''Move a file, using /bin/cp via sudo if requested. Will work around known bugs in python.'''
    from shutil import move
    from os import remove, path
    
    result = FslIResult("", FslIResult.SUCCESS, '')
    
    if requires_root:
        cmd_result = run_cmd(" ".join(("/bin/cp",from_file,to_file)), as_root=True)
        if cmd_result.status == FslIResult.ERROR:
            MsgUser.debug(cmd_result.message)
            result = FslIResult('', FslIResult.ERROR, "Failed to move %s (%s)" % (from_file, cmd_result.message))
        else:
            remove(from_file)
    else:
        try:
            move(from_file, to_file)
        except OSError, e:
            # Handle bug in some python versions on OS X writing to NFS home folders
            # Python tries to preserve file flags but NFS can't do this. It fails to catch
            # this error and ends up leaving the file in the original and new locations! 
            if e.errno == 45:
                # Check if new file has been created:
                if path.isfile(to_file):
                    # Check if original exists
                    if path.isfile(from_file):
                        # Destroy original and continue
                        remove(from_file)
                else:
                    cmd_result = run_cmd("/bin/cp %s %s" % (from_file, to_file), as_root=False)
                    if cmd_result.status == FslIResult.ERROR:
                        MsgUser.debug(cmd_result.message)
                        result = FslIResult('', FslIResult.ERROR, "Failed to copy from %s (%s)" % (from_file, cmd_result.message))
                    remove(from_file)
            else:
                raise
        except:
            raise
    return result

def edit_file(fname, edit_function, search_for, replace_with, requires_root):
    '''Search for a simple string in the file given and replace it with the new text'''
    from os import remove
    result =  FslIResult('', FslIResult.SUCCESS, '')

    try:
        (tmpfile, tmpfname) = tempFileName(mode='w')
        src = open(fname)
    
        for line in src:
            line = edit_function(line, search_for, replace_with)
            tmpfile.write(line)
        src.close()
        tmpfile.close()

        try:
            cmd_result = move_file(tmpfname, fname, requires_root)
            if cmd_result.status == FslIResult.ERROR:
                MsgUser.debug(cmd_result.message)
                result = FslIResult('', FslIResult.ERROR, "Failed to edit %s (%s)" % (fname, cmd_result.message))
        except:
            remove(tmpfname)
            raise
    except IOError, e:
        MsgUser.debug(e.strerror)
        result = FslIResult('', FslIResult.ERROR, "Failed to edit %s" % (fname))
    MsgUser.debug("Modified %s (search %s; replace %s)." % (fname, search_for, replace_with))
    return result

def add_to_file(fname, add_lines, requires_root):
    '''Add lines to end of a file'''
    from os import remove
    result = FslIResult('', FslIResult.SUCCESS, '')

    try:
        (tmpfile, tmpfname) = tempFileName(mode='w')
        src = open(fname)
    
        for line in src:
            tmpfile.write(line)
        src.close()
        tmpfile.write('\n')
        for line in add_lines:
            tmpfile.write(line)
            tmpfile.write('\n')
        tmpfile.close()
        try:
            cmd_result = move_file(tmpfname, fname, requires_root)
            if cmd_result.status == FslIResult.ERROR:
                MsgUser.debug(cmd_result.message)
                result = FslIResult('', FslIResult.ERROR, "Failed to add to file %s (%s)" % (fname, cmd_result.message))
        except:
            remove(tmpfname)
            raise
    except IOError, e:
        MsgUser.debug(e.strerror + tmpfname + fname)
        result = FslIResult('', FslIResult.ERROR, "Failed to add to file %s" % (fname))
    MsgUser.debug("Modified %s (added %s)" % (fname, '\n'.join(add_lines)))
    return result

def create_file(fname, lines, requires_root):
    '''Create a new file containing lines given'''
    from os import remove
    result = FslIResult('', FslIResult.SUCCESS, '')
    try:
        (tmpfile, tmpfname) = tempFileName(mode='w')
    
        for line in lines:
            tmpfile.write(line)
            tmpfile.write('\n')
        tmpfile.close()
        try:
            cmd_result = move_file(tmpfname, fname, requires_root)
            if cmd_result.status == FslIResult.ERROR:
                MsgUser.debug(cmd_result.message)
                result = FslIResult('', FslIResult.ERROR, "Failed to edit %s (%s)" % (fname, cmd_result.message))
        except:
            remove(tmpfname)
            raise
    except IOError, e:
        MsgUser.debug(e.strerror)
        result = FslIResult('', FslIResult.ERROR, "Failed to create %s" % (fname))
    MsgUser.debug("Created %s (added %s)" % (fname, '\n'.join(lines)))
    return result

def find_X11():
    '''Function to find X11 install on Mac OS X and confirm it is compatible. Advise user to download Xquartz if necessary'''
    from os import path
    from subprocess import Popen,PIPE, STDOUT
    
    MsgUser.message("Checking for X11 windowing system (required for FSL GUIs).")
    bad_versions=[]
    xquartz_url = 'http://xquartz.macosforge.org/landing/'
    xbins = [ 'XQuartz.app', 'X11.app' ]
    xloc = '/Applications/Utilities'
    xbin = ''
    
    for x in xbins:
        if path.exists('/'.join((xloc,x))):
            xbin = x
    
    if xbin != '':
        # Find out what version is installed
        x_v_cmd = [ '/usr/bin/mdls', '-name', 'kMDItemVersion', '/'.join((xloc, xbin)) ] 
        cmd = Popen(x_v_cmd, stdout=PIPE, stderr=STDOUT)
        (vstring,_) = cmd.communicate()
        
        if cmd.returncode:
            MsgUser.debug("Error finding the version of X11 (%s)" % (vstring))
            # App found, but can't tell version, warn the user
            result = FslIResult(0, FslIResult.WARN, "X11 (required for FSL GUIs) is installed but I can't tell what the version is.")
        else:
            # Returns:
            # kMDItemVersion = "2.3.6"\n
            (_,_,version) = vstring.strip().split()
            if version.startswith('"'):
                version = version[1:-1]
            if version in bad_versions:
                result = FslIResult(0, FslIResult.WARN, "X11 (required for FSL GUIs) is a version that is known to cause problems. We suggest you upgrade to the latest XQuartz release from %s" % (xquartz_url))
            else:
                MsgUser.debug("X11 found and is not a bad version (%s: %s)." %(xbin, version))
                result = FslIResult(0, FslIResult.SUCCESS, 'X11 software (required for FSL GUIs) is installed.')
    else:
        # No X11 found, warn the user
        result = FslIResult(0, FslIResult.WARN, "The FSL GUIs require the X11 window system which I can't find in the usual places. You can download a copy from %s - you will need to install this before the GUIs will function" % (xquartz_url))
    return result


class UnsupportedOs(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
class Os(object):
    '''Work out which platform we are running on'''

    def __init__(self):
        import os
        if os.name != 'posix': raise UnsupportedOs('We only support OS X/Linux')
        import platform
        self.os = platform.system().lower()
        self.arch = platform.machine()
        self.applever = ''
        
        if self.os == 'darwin':
            self.vendor = 'apple'
            self.version = Version(platform.release())
            (self.applever,_,_) = platform.mac_ver()
            if self.arch == 'Power Macintosh': raise UnsupportedOs('We no longer support PowerPC')
            self.glibc = ''
            self.bits = ''
        elif self.os == 'linux':
            if hasattr(platform, 'linux_distribution'):
                # We have a modern python (>2.4)
                (self.vendor, version, _) = platform.linux_distribution(full_distribution_name=0)
            else:
                (self.vendor, version, _) = platform.dist()
            self.vendor = self.vendor.lower()
            self.version = Version(version)
            self.glibc = platform.libc_ver()
            if self.arch == 'x86_64':
                self.bits = '64'
            else:
                self.bits = '32'
                raise UnsupportedOs("We no longer support 32 bit Linux. If you must use 32 bit Linux then try building from our sources.")
        else:
            raise UnsupportedOs("We don't support this OS, you should try building from our sources.")
        # Now check for supported OS
    
def is_writeable(location):
    '''Check if we can write to the location given'''
    import tempfile, errno
 
    writeable = True
    try:
        tfile = tempfile.NamedTemporaryFile(mode='w+b', dir=location)
        tfile.close()
    except OSError,e:
        if e.errno == errno.EACCES:
            writeable = False
        else:
            raise
    return writeable

def is_writeable_as_root(location):
    '''Check if sudo can write to a given location'''
    # This requires us to use sudo
    from os import remove, path
    
    (f, fname) = tempFileName(mode='w')
    f.write("FSL")
    f.close()
    
    result = False
    tmptarget = '/'.join((location, path.basename(fname)))
    MsgUser.debug( " ".join(('/bin/cp', fname, tmptarget)) )
    cmd_result = run_cmd(" ".join(('/bin/cp', fname, tmptarget)), as_root=True)
    if cmd_result.status == FslIResult.SUCCESS:
        result = True
        remove(fname)
        cmd_result = run_cmd(" ".join(('/bin/rm', '-f', tmptarget)), as_root=True)
    else:
        MsgUser.debug( cmd_result.message )
        remove(fname)
        result = False
    MsgUser.debug("Writeable as root? %s" % (result))
    return result
    
def md5File(filename):
    '''Returns the MD5 sum of the given file.'''
    try:
        import hashlib
        fhash = hashlib.md5()
        bs = fhash.block_size
    except ImportError:
        import md5
        fhash = md5.new()
        # For efficiency read file in 8K blocks
        bs = 8192
    f = open(filename,'rb')
    while True:
        nextblk = f.read(bs)
        if not nextblk:
            break
        fhash.update(nextblk)
    f.close()
    return fhash.hexdigest()

def parsemd5sumfile(md5string):
    '''Returns MD5 sum extracted from the output of md5(sum) from OS X/Linux platforms'''
    if md5string.startswith('MD5'):
        # OSX format
        (_,_,_,md5) = md5string.split()
    else:
        (md5,_) = md5string.split()
    return md5
    
      
def checkmd5Sum(filename, md5F):    
    '''Cross-platform verification of md5 sum. Takes file to check and file containing the output of md5(sum) on your platform for the file'''

    fileMD5 = md5File(filename)
    f = open(md5F)
    md5sum = f.readline().strip()
    f.close()
    
    # Linux has format MD5 (*)filename
    # OS X  has format MD5 (filename) = MD5
    
    md5 = parsemd5sumfile(md5sum)
    MsgUser.debug( "Comparing: %s to %s" % (fileMD5, md5))
    return fileMD5 == md5

def open_url(url, start=0, timeout=20):
    import urllib2
    import socket
    socket.setdefaulttimeout(timeout)
    MsgUser.debug("Attempting to download %s." % (url))
    
    try:
        req = urllib2.Request(url)
        if start != 0:
            req.headers['Range'] = 'bytes=%s-' % (start)
        rf = urllib2.urlopen(req)
    except urllib2.HTTPError, e:
        MsgUser.debug("%s %s" % (url, e.msg))
        return FslIResult(False, FslIResult.ERROR, "Cannot find file %s on server (%s). Try again later." % (url, e.msg))
    except urllib2.URLError, e:
        errno = e.reason.args[0]
        message = e.reason.args[1]
        if errno == 8:
            # Bad host name
            MsgUser.debug("%s %s" % (url, 'Unable to find FSL download server in the DNS'))
        else:
            # Other error
            MsgUser.debug("%s %s" % (url, message))
        return FslIResult(False, FslIResult.ERROR, "Cannot find %s (%s). Try again later." % (url, message))
    except socket.timeout, e:
        MsgUser.debug(e)
        return FslIResult(False, FslIResult.ERROR, "Failed to contact FSL web site. Try again later.")
    return FslIResult(rf, FslIResult.SUCCESS,'')
    
def download_file(url, localf, timeout=20, tui=False):
    '''Get a file from the url given storing it in the local file specified'''
    import socket, time
    
    result = open_url(url, 0, timeout)

    if result.status == FslIResult.SUCCESS:
        rf = result.result
    else:
        return result
    
    metadata = rf.info()
    rf_size = int(metadata.getheaders("Content-Length")[0])
    
    dl_size = 0
    block = 16384
    if tui:
        (y,x) = tui.getyx()
    else:
        x = 0
        y= 0
    pb = Progress_bar( tui, x, y, rf_size, numeric=True)

    for attempt in range(1,6):
        # Attempt download 5 times before giving up
        pause = timeout
        try:  
            try:
                lf = open(localf, 'ab')    
            except:
                return FslIResult(False, FslIResult.ERROR, "Failed to create temporary file.")

            while True:
                buf = rf.read(block)
                if not buf:
                    break
                dl_size += len(buf)
                lf.write(buf)
                pb.update( dl_size )
            lf.close()
        except (IOError, socket.timeout), e:
            MsgUser.debug(e.strerror)
            MsgUser.message("\nDownload failed re-trying (%s)..." % attempt)
            pause = 0
        if dl_size != rf_size:
            time.sleep(pause)
            MsgUser.message("\nDownload failed re-trying (%s)..." % attempt)
            result = open_url(url, dl_size, timeout)
            if result.status == FslIResult.ERROR:
                MsgUser.debug(result.message)
            else:
                rf = result.result
        else:
            break      
    if dl_size != rf_size:
        return FslIResult(False, FslIResult.ERROR, "Failed to download file.")
    return FslIResult(True, FslIResult.SUCCESS, '')   

def getAndReportMD5(url):
    from os import remove
    (_, md5fname) = tempFileName(mode='w',close=True)
    
    a_result = download_file(url, md5fname)
    if a_result.status == FslIResult.SUCCESS:
        MD5file = open(md5fname)
        MD5sum = MD5file.readline().strip()
        MD5file.close()
        remove(md5fname)
        MsgUser.message("MD5 checksum is: %s" % (parsemd5sumfile(MD5sum)))
    else:
        MsgUser.debug("Failed to download checksum.")
    return
    
def getAndVerifyMD5(url, checkF):
    from os import remove
    
    result = FslIResult(True, FslIResult.SUCCESS, '')
    
    (_,md5fname) = tempFileName(mode='w',close=True)
    
    a_result = download_file(url, md5fname)
    if a_result.status == FslIResult.SUCCESS:
        if not checkmd5Sum(checkF, md5fname):
            MsgUser.debug("Failed to verify %s." % (checkF))
            result = FslIResult(False, FslIResult.ERROR, 'File %s appears to be corrupt.' % (checkF))
        else:
            MsgUser.debug("%s verified." % (checkF))
            result = FslIResult(True, FslIResult.SUCCESS, 'File verified.')
    else:
        MsgUser.debug("Unable to download checksum for %s" % (checkF))
        result = FslIResult(False, FslIResult.WARN, 'Unable to download checksum for %s.' % (checkF))
    remove(md5fname)
    return result

def fastest_mirror(main_site='http://fsl.fmrib.ox.ac.uk/fsldownloads/',
                   mirror_file='fslmirrorlist.txt',
                   timeout=20,
                   serverport=80):
    '''Find the fastest mirror for FSL downloads.'''
    import urllib2
    import socket
    from time import time
    MsgUser.debug("Calculating fastest mirror")
    socket.setdefaulttimeout(timeout)
    download_url=main_site
    if not main_site.endswith('/'):
        main_site += '/'
    mirror_url=main_site + mirror_file
    # Get the mirror list from the url
    fastestmirrors = {}
    
    try:
        response = urllib2.urlopen(url=mirror_url)
    except urllib2.HTTPError, e:
        MsgUser.debug("%s %s" % (mirror_url, e.msg))
        return FslIResult('', FslIResult.ERROR, "Failed to download mirror list (%s). Try again later." % (e.msg))
    except urllib2.URLError, e:
        if isinstance(e.reason, socket.timeout):
            MsgUser.debug("Time out trying %s" % (mirror_url))
        else:
            MsgUser.debug(e.reason.args[1])
        return FslIResult('', FslIResult.ERROR, "Failed to find main FSL web site. Try again later.")
    except socket.timeout, e:
        MsgUser.debug(e)
        return FslIResult('', FslIResult.ERROR, "Failed to contact FSL web site. Try again later.")
    except:
        raise
    mirrorlist = response.read().strip().split('\n')
    MsgUser.debug("Received the following mirror list %s" % (mirrorlist))
    if len(mirrorlist) == 0:
        mirrorlist[0]=download_url

    try:
        servers = dict([ (k.split('/')[2], k.split('/',3)[3]) for k in mirrorlist])
    except IndexError:
        return FslIResult('', FslIResult.ERROR, "Failed to get a valid list of FSL mirror sites. Are you connected to the internet?")
    # Check timings from the urls specified
    if len(servers) > 1:
        for mirror in servers.keys():
            MsgUser.debug( "Trying %s" % (mirror) )
            then = time()

            try:
                mysock=socket.create_connection((mirror, serverport), timeout)
                pingtime = time() - then
                mysock.close()
                fastestmirrors[pingtime] = mirror
                MsgUser.debug("Mirror responded in %s seconds" % (pingtime))
            except socket.gaierror, e:
                MsgUser.debug("%s can't be resolved" % (e))
            except socket.timeout, e:
                MsgUser.debug(e)
        if len(fastestmirrors) == 0:
            return FslIResult('', FslIResult.ERROR, 'Failed to contact any download sites.')
        host = fastestmirrors[min(fastestmirrors.keys())]
        download_url = '/'.join(('http:/', host, servers[host]))
    else:
        download_url = mirrorlist[0]
    
    return FslIResult(download_url, FslIResult.SUCCESS, 'Success')


    
    
def get_web_version(download_url, timeout=20):
    '''Download the file given by download_url and return the version info held within'''
    import urllib2
    import socket
    socket.setdefaulttimeout(timeout)
    MsgUser.debug("Looking for latest version at %s." % (download_url))
    try:
        response = urllib2.urlopen(url=download_url)
    except urllib2.HTTPError, e:
        MsgUser.debug("%s %s" % (download_url, e.msg))
        return FslIResult(False, FslIResult.ERROR, "Cannot find file %s on server (%s). Try again later." % (download_url, e.msg))
    except urllib2.URLError, e:
        errno = e.reason.args[0]
        message = e.reason.args[1]
        if errno == 8:
            # Bad host name
            MsgUser.debug("%s %s" % (download_url, 'Unable to find FSL download server in the DNS'))
        else:
            # Other error
            MsgUser.debug("%s %s" % (download_url, message))
        return FslIResult(Version('0.0.0'), FslIResult.ERROR, "Failed to find FSL web site. Try again later.")
    
    try:
        v_line = response.read().strip()
        server_v = Version(v_line)
        MsgUser.debug("Found a version number of %s." % (server_v))
        result = FslIResult(server_v, FslIResult.SUCCESS, 'Success')
    except ValueError, e:
        MsgUser.debug(str(e) + " %s\n" % (v_line))
        result = FslIResult(Version('0.0.0'), FslIResult.ERROR, 'Server reported bad version (%s)' %(v_line))
    return result

def get_fsldir():
    '''Find the installed version of FSL using FSLDIR or location of this script'''
    from os import path, environ
    
    try:
        fsldir = environ['FSLDIR']
        if not path.exists( fsldir ):
            # FSLDIR environment variable is incorrect!
            MsgUser.warning('FSLDIR environment variable is corrupt, ignoring...')
            MsgUser.debug('FSLDIR is set to %s - this folder does not appear to exist' % (fsldir) )
            fsldir = ''
    except KeyError:
        # FSLDIR not set for this user's environment - try to find FSL
        # Installer should be in FSLDIR/etc
        fsldir = path.dirname(path.dirname( path.realpath( __file__ )))
        v_file = "/".join((fsldir,'etc/fslversion'))
        if not path.exists( v_file ):
            # No FSL install found
            fsldir = ''
    return fsldir

def tarf_version(tarfile):
    '''Takes the path to a tar(gz) FSL install file and works out what version it is.'''
    from os import path
    
    if not path.isfile(tarfile):
        return FslIResult(Version('0.0.0'), FslIResult.ERROR, "%s is not a file" % (tarfile))
    else:
        # file is of form: fsl-V.V.V-platform.tar.gz
        (_,vstring,_) = tarfile.strip().split('-')
        return FslIResult(Version(vstring), FslIResult.SUCCESS,'')
    
     
def get_installed_version(fsldir):
    '''Takes path to FSLDIR and finds installed version details'''
    from os import path
    MsgUser.debug("Looking for fsl in %s" % fsldir)
    v_file = "/".join((fsldir,'etc/fslversion'))
    if path.exists( v_file ):
        f = open(v_file)
        v_string = f.readline()
        f.close()
        result = FslIResult(Version(v_string.strip()), FslIResult.SUCCESS, "Success")
    else:
        result = FslIResult(Version('0.0.0'), FslIResult.ERROR, "No FSL install found")
    MsgUser.debug("Found version %s" % (result.result)) 
    return result

def self_update(mirror_url, scr=False):
    '''Check for and apply an update to myself'''
    from os import execv, chmod
    from sys import argv
    
    # See if there is a newer version available

    url = "/".join((mirror_url.rstrip('/'), 'fslinstaller.py'))
    web_version = get_web_version('-'.join((url, "version.txt"))).result
    if web_version <= version:
        MsgUser.debug("Installer is up-to-date." )
        return FslIResult(True, FslIResult.SUCCESS, 'Installer is current')
    
    # There is a new version available - download it
    MsgUser.message("There is a newer version (%s) of the installer (you have %s) updating..." % (web_version, version))
    (_, tmpfname) = tempFileName(mode='w', close=True)
    
    a_result = download_file(url, tmpfname)
    if a_result.status == FslIResult.SUCCESS:
        url = "/".join((mirror_url.rstrip('/'), 'md5sums', 'fslinstaller.py.md5'))
        # Got the new version, now get the MD5 of the file
        MsgUser.debug("Downloading %s to check %s" % (url,tmpfname))
        m_result = getAndVerifyMD5(url, tmpfname)
        if m_result.status != FslIResult.SUCCESS:
            return FslIResult(False, FslIResult.ERROR, m_result.message)
    else:
        MsgUser.debug("Failed to update installer %s." % (a_result.message))
        return FslIResult(False, FslIResult.ERROR, 'Found update to installer but unable to download the new version. Please try again.')
    # Now run the new installer
    # EXEC new script with the options we were given
    chmod(tmpfname, 0555)
    execv(tmpfname, argv)

def which_shell():
    from os import getenv
    return getenv("SHELL").split("/")[-1]
    
class FslInstall(object):
    shells = {}
    shells['bash'] = '.bash_profile'
    shells['sh'] = '.profile'
    shells['ksh'] = '.profile'
    shells['csh'] = '.cshrc'
    shells['tcsh'] = '.cshrc'
    
    shellmap = {}
    shellmap['bash'] = 'sh'
    shellmap['ksh'] = 'sh'
    shellmap['tcsh'] = 'csh'
    shellmap['sh'] = 'sh'
    shellmap['csh'] = 'csh'
    
    tar_roots = {}
    tar_roots['redhat'] = 'centos'
    tar_roots['centos'] = 'centos'
    tar_roots['apple'] = 'macosx'
    
    supported = {}
    supported['darwin'] = Platform('darwin')
    supported['darwin'].addVendor('apple')
    
    maps = {}
    unsupported = {}
    unsupported['darwin'] = Platform('darwin')
    unsupported['darwin'].addVendor('apple')
    for release in range(10,15):
        supported['darwin'].vendors['apple'].releases.append(release)
    unsupported['darwin'].vendors['apple'].releases.append(15)
    
    supported['linux'] = Platform('linux')
    unsupported['linux'] = Platform('linux')
    
    for vendor in ('centos', 'redhat'):
        supported['linux'].addVendor(vendor)
        unsupported['linux'].addVendor(vendor)
        for release in range(5,7):
            supported['linux'].vendors[vendor].releases.append(release)
        unsupported['linux'].vendors[vendor].releases.append(7)
        # Until official support install Centos 6 version on Centos 7
        maps[vendor] = {}
        maps[vendor][7] = 6
    unsupported['linux'].addVendor('fedora')
    maps['fedora'] = {}
    for release in range(13,22):
        unsupported['linux'].vendors['fedora'].releases.append(release)
        if release > 18 and 7 in supported['linux'].vendors['centos'].releases:
            maps['fedora'][release] = 7
        else:
            maps['fedora'][release] = 6
    license = '''
    FSL License goes here
    
    '''
    def __init__(self, fsldir, version, mirror, keep, shell):
        from os import getuid, path
        self.lines = {}
        self.test = {}
        self.skiproot_head = {}
        self.skiproot_foot = {}
        self.fsldir = fsldir
        self.location = path.dirname(fsldir)
        self.version = version
        self.shell = shell
        try:
            self.profile = "/".join((path.expanduser("~"), self.__class__.shells[shell]))
        except KeyError:
            raise UnsupportedOs("This installer doesn't support this shell (%s). You will have to manually install." % (shell))
        self.computer = Os()
        self.keep = keep
        self.mirror = mirror
        
        if self.computer.vendor in self.unsupported['linux'].vendors.keys():
            if self.computer.version.major >= self.unsupported['linux'].vendors[self.computer.vendor].releases[0]:
                # We don't officially support Fedora, but things should work OK with the Centos 6 binaries for Fedora > 13
                MsgUser.warning("Unsupported OS (%s %s %s.%s) - will try to install nearest equivalent FSL version." % ( self.computer.vendor, self.computer.os,
                                                                 self.computer.version.major, self.computer.version.minor ))
                MsgUser.warning("You may need to install libpng-compat (sudo yum install libpng-compat) to use FSLView. If you have any issues please compile FSL from the source code." )
                self.computer.version.major = self.maps[self.computer.vendor][self.computer.version.major]
                self.computer.vendor = 'centos'

        supported_os = self.supported
        unsupported_os = self.unsupported
        if self.computer.vendor not in supported_os[self.computer.os].vendors.keys():
            
            raise UnsupportedOs("This installer doesn't support this OS (%s %s %s.%s). Debian/Ubuntu users should look on the NeuroDebian site for installation instructions. Other OSes may need to compile FSL from our source code." %
                                (self.computer.vendor, self.computer.os, 
                                 self.computer.version.major, self.computer.version.minor))
 
        if self.computer.os not in supported_os.keys() or self.computer.version.major not in supported_os[self.computer.os].vendors[self.computer.vendor].releases:
            MsgUser.debug("Unsupported OS (%s %s %s) - is there a fall back?" % (self.computer.vendor,
                                                                                 self.computer.os,
                                                                                 self.computer.version.major))
            MsgUser.debug(unsupported_os[self.computer.os].vendors[self.computer.vendor].releases)
            if self.computer.version.major in unsupported_os[self.computer.os].vendors[self.computer.vendor].releases:
                MsgUser.warning("Unsupported OS (%s %s %s.%s). This version of the OS has not been fully tested." % (self.computer.vendor,
                                                                                                                    self.computer.os,
                                                                                                                    self.computer.version.major,
                                                                                                                    self.computer.version.minor))
                if self.computer.vendor == 'centos' or self.computer.vendor == 'redhat':
                    self.computer.version.major = supported_os['linux'].vendors[self.computer.vendor].releases[-1]
                elif self.computer.vendor == 'apple':
                    self.computer.version.major = supported_os['darwin'].vendors[self.computer.vendor].releases[-1]
            else:
                raise UnsupportedOs("This installer doesn't support this OS (%s %s %s.%s). Debian/Ubuntu users should look on the NeuroDebian site for installation instructions." %
                                    (self.computer.vendor, self.computer.os, 
                                     self.computer.version.major, self.computer.version.minor))
        
        tar_roots = self.__class__.tar_roots
        if self.computer.vendor == 'apple':
            platform_string = tar_roots[self.computer.vendor]
        else:
            platform_string = '_'.join( (tar_roots[self.computer.vendor]+str(self.computer.version.major), self.computer.bits) )
            
        self.tarf = '-'.join( ('fsl',
                               str(self.version),
                               platform_string + '.tar.gz' ) )
        
        self.url = '/'.join((self.mirror.rstrip('/'), self.tarf))
        self.lines['sh'] = []
        self.lines['sh'].append("# FSL Setup")
        self.lines['sh'].append('FSLDIR=%s' % (self.fsldir))
        self.lines['sh'].append('PATH=${FSLDIR}/bin:${PATH}')
        self.lines['sh'].append('export FSLDIR PATH')
        self.lines['sh'].append('. ${FSLDIR}/etc/fslconf/fsl.sh')
        self.test['sh'] = 'FSLDIR='
        self.skiproot_head['sh'] = []
        for line in ( 'if [ -x /usr/bin/id ]; then',
                      'if [ -z "$EUID" ]; then',
                      "# ksh and dash doesn't setup the EUID environment var",
                      'EUID=`id -u`',
                      'fi',
                      'fi',
                      'if [ "$EUID" != "0" ]; then' ):
            self.skiproot_head['sh'].append(line)
        self.skiproot_foot['sh'] = 'fi'
                
        self.lines['csh'] = []
        self.lines['csh'].append("# FSL Setup")
        self.lines['csh'].append('setenv FSLDIR %s' % (self.fsldir))
        self.lines['csh'].append('setenv PATH ${FSLDIR}/bin:${PATH}')
        self.lines['csh'].append('source ${FSLDIR}/etc/fslconf/fsl.csh')     
        self.test['csh'] = 'setenv FSLDIR '
        self.skiproot_head['csh'] = []
        self.skiproot_head['csh'].append( 'if ( $uid != 0 ) then' )
        self.skiproot_foot['csh'] = 'endif'
        
        self.lines['startup.m'] = []
        self.lines['startup.m'].append("setenv( 'FSLDIR', '%s' );" % (self.fsldir))
        self.lines['startup.m'].append("fsldir = getenv('FSLDIR');")
        self.lines['startup.m'].append("fsldirmpath = sprintf('%s/etc/matlab',fsldir);")
        self.lines['startup.m'].append("path(path, fsldirmpath);")
        self.lines['startup.m'].append("clear fsldir fsldirmpath;")
        self.test['startup.m'] = "setenv( 'FSLDIR', "

        self.sudo = False
        if getuid() != 0:
            self.sudo = True
        
      
    def patch_systemprofile(self):
        '''Add a system-wide profile setting up FSL for all users. Only supported on Redhat/Centos'''
        from os import path
        profile_d = '/etc/profile.d'
        profile_files = ['fsl.sh', 'fsl.csh']
        if path.isdir(profile_d):
            for profile in profile_files:
                pf = profile.split('.')[1]
                lines = self.lines[pf]
                perfect_match = self.lines[pf][0]
                strfind = self.test[pf]
                this_profile = '/'.join((profile_d, profile))
                if path.exists(this_profile):
                    # Already has a profile file
                    # Does it contain an exact match for current FSLDIR?
                    match = file_contains_1stline(this_profile, perfect_match)
                    if match != '':
                        # If there is an fsl.(c)sh then just fix the entry for FSLDIR
                        MsgUser.debug( "Fixing %s for FSLDIR location." % (this_profile))
                        result = edit_file(this_profile, line_starts_replace, strfind, perfect_match, self.sudo)
                    else:
                        # No need to do anything
                        MsgUser.debug( "%s already configured - skipping." %(this_profile))
                        result = FslIResult(0, FslIResult.SUCCESS, "Skipped")
                else:
                    # Create the file
                    result = create_file(profile, lines, self.sudo)
        else:
            result = FslIResult(0, FslIResult.WARN, "No system-wide configuration folder found - Skipped")
        return result
    
    def fix_fsldir(self):
        try:
            search_str=self.test[self.__class__.shellmap[self.shell]]
            # First line is a comment...
            replace_str=self.lines[self.__class__.shellmap[self.shell]][1]
        except KeyError:
            MsgUser.debug("Fixing FSLDIR entry: Unknown shell (%s)." %(self.shell))
            return FslIResult(False, FslIResult.ERROR, 'Unknown shell')
        MsgUser.debug("Editing %s, replacing line:%s with %s." % (self.profile, search_str, replace_str))
        result = edit_file(self.profile, line_starts_replace, search_str, replace_str, False)
        return result
    
    def add_fsldir(self):
        try:
            fsl_env = self.lines[self.__class__.shellmap[self.shell]]
        except KeyError:
            MsgUser.debug("Adding FSLDIR entry: Unknown shell (%s)" %(self.shell))
            return FslIResult(False, FslIResult.ERROR, 'Unknown shell')
        return add_to_file(self.profile, fsl_env, False)   

    def configure_matlab(self, m_startup='', c_file=True):
        '''Setup your startup.m file to enable FSL MATLAB functions to work'''
        from os import path, getenv, mkdir

        mlines = self.lines['startup.m']
        if m_startup == '':
            m_startup = '/'.join((getenv('HOME'), 'matlab', 'startup.m'))
        if path.exists(m_startup):
            # Check if already configured
            match = file_contains_1stline(mlines[0], m_startup)
            MsgUser.debug(match)
            if match == '':
                match = file_contains(m_startup, self.test['startup.m'])
                if match:
                    result = edit_file(m_startup, line_starts_replace, self.test['startup.m'], mlines[0], False)
                    MsgUser.debug('MATLAB startup file is miss-configured, replacing %s with %s.' % (self.test['startup.m'], mlines[0]))
                else:
                    MsgUser.debug('Adding FSL settings to MATLAB.')
                    result = add_to_file(m_startup, mlines, False)
            else:
                MsgUser.debug('MATLAB already configured.')
                result = FslIResult(True, FslIResult.WARN, "MATLAB already configured.")
        elif c_file:
            # No startup.m file found. Create one
            try:
                MsgUser.debug('No MATLAB startup.m file found, creating one.')
                if not path.isdir( path.dirname(m_startup)):
                    MsgUser.debug('No MATLAB startup.m file found, creating one.')
                    mkdir(path.dirname(m_startup))
                result = create_file(m_startup, mlines,False)    
            except OSError:
                MsgUser.debug('Unable to create ~/matlab folder, cannot configure.')
                result = FslIResult(False, FslIResult.ERROR, "Unable to create your ~/matlab folder, so cannot configure MATLAB for FSL.")
        else:
            MsgUser.debug('MATLAB may not be installed, doing nothing.')
            result = FslIResult(True, FslIResult.WARN, "I can't tell if you have MATLAB installed.")
        return result
    
    def slash_applications_links(self):
        '''Create symlinks to FSLView (and qassistant) in /Applications'''
        from os import path
        
        MsgUser.message("Creating Application links...")
        results = {}
        apps = ['fslview.app', 'assistant.app' ]
        for app in apps:
            app_location = '/'.join(('/Applications', app))
            app_target = '/'.join((self.location, 'fsl', 'bin', app))
            create_link = False
            MsgUser.debug("Looking for existing link %s" % (app_location) )
            if path.lexists( app_location ):
                MsgUser.debug("Is a link: %s; realpath: %s" % (path.islink(app_location), path.realpath(app_location)))
                if path.islink( app_location ):
                    MsgUser.debug("A link already exists.")
                    if path.realpath( app_location ) != app_target:
                        MsgUser.debug("Deleting old (incorrect) link %s" % (app_location))
                        result = run_cmd("rm " + app_location, as_root=self.sudo)
                        if result.status == FslIResult.SUCCESS:
                            create_link = True
                        else:
                            MsgUser.debug("Unable to remove broken link to %s (%s)." % (app_target, result.message))
                            results[app] = FslIResult(False, FslIResult.WARN, 'Unable to remove broken link to %s' % (app_target))
                    else:
                        MsgUser.debug("Link is correct, skipping.")
                else:
                    MsgUser.debug("%s doesn't look like a symlink, so let's not delete it." % (app_location))
                    results[app] = FslIResult(False, FslIResult.WARN, "%s is not a link, I won't delete it. You" % (app_location))
            else:
                MsgUser.debug('Create a link for %s' % (app))
                create_link = True
            MsgUser.debug('Looking for %s (create_link is %s)' %(app_target, create_link))
            if create_link:
                if path.exists( app_target ):
                    result = run_cmd("ln -s %s %s" % (app_target, app_location ), as_root=self.sudo)
                    if result.status != FslIResult.SUCCESS:
                        MsgUser.debug("Unable to create link to %s (%s)" % (app_target, result.message))
                        results[app] = FslIResult(False, FslIResult.WARN, 'Unable to remove broken link to %s' % (app_target))
                else:
                    results[app] = FslIResult(False, FslIResult.WARN, 'Unable to find application %s to link to' % (app_target))
            else:
                results[app] = FslIResult(True, FslIResult.WARN, "Application %s already linked." % (app))
        return results
        
    def configure_env(self):
        '''Setup the user's environment so that their terminal finds the FSL tools etc.'''
        from os import path
        # Check for presence of profile file:
        cfile = False
        if not path.isfile(self.profile):
            MsgUser.debug("User is missing a shell setup file.")
            if self.__class__.shells['bash'] in self.profile:
                # Check if they have a .profile instead
                dot_profile = "/".join((path.expanduser("~"), self.__class__.shells['sh']))
                if path.isfile(dot_profile):
                    self.profile = dot_profile
                else:
                    cfile = True
            else:
                cfile = True
        if cfile:
            MsgUser.debug("Creating file %s with content %s" % (self.profile, self.lines[self.shellmap[self.shell]]))
            a_result = create_file(self.profile, self.lines[self.shellmap[self.shell]], False)
        else:
            # Check if user already has FSLDIR set
            MsgUser.message("Setting up FSL software...")
            if file_contains(self.profile, "FSLDIR"):
                MsgUser.debug("Updating FSLDIR entry.")
                a_result = self.fix_fsldir()
            else:
                MsgUser.debug("Adding FSLDIR entry.")
    
                a_result = self.add_fsldir()
                if a_result.status == FslIResult.ERROR:
                    MsgUser.debug(a_result.message)
                    return FslIResult('', FslIResult.ERROR, "Failed to update your profile %s with FSL settings" % (self.shell))
        # Mac OS X machines need MATLAB setting up too
        if self.computer.vendor == 'apple':
            a_result = self.configure_matlab()
            if a_result.status == FslIResult.ERROR:
                MsgUser.debug(a_result.message)
                return a_result
            elif a_result.status == FslIResult.WARN:
                MsgUser.skipped(a_result.message)
        return FslIResult('', FslIResult.SUCCESS, "Profile configured.")

    def install_installer(self):
        '''Install this script into $FSLDIR/etc'''
        from os import path
        targetfolder = "/".join((self.fsldir.rstrip('/'), "etc"))
        as_root = False
        installer = path.abspath(__file__)
        MsgUser.debug("Copying fslinstaller (%s) to %s" %(installer, targetfolder))
        try:
            if not is_writeable(targetfolder):
                if not is_writeable_as_root(targetfolder):
                    raise InstallFailed("Cannot write to folder as root user.")
                else:
                    as_root = True
        except InstallFailed, e:
            MsgUser.debug(e.value)
            return FslIResult('', FslIResult.ERROR, "Installer updated, but cannot be copied to existing FSL install.")
        else:
            return copy_file(installer, "/".join((targetfolder, "fslinstaller.py")), as_root )

    def install_tar(self, targz_file):
        '''Install given tar.gz file, use sudo where required'''
        #import tarfile
        from os import getpid,path
        from time import time
        # Check if install location is writable
        MsgUser.debug("Checking %s is writeable." % (self.location))
        if is_writeable(self.location):
            as_root = False
        elif is_writeable_as_root(self.location):
            as_root = True
        else:
            return FslIResult(False, FslIResult.ERROR, "Unable to write to target folder (%s)" % (self.location))

        MsgUser.message("Installing FSL software...")
        # Find out how many files to install
        #tarf = tarfile.open(targz_file, mode='r:gz')
        #MsgUser.debug("Opened the tar file for reading...")
        #tarcontents = tarf.getnames()
        #nfiles = len(tarcontents)
        #tarf.close()
        #MsgUser.debug("Calculated the number of install objects as %s" % (nfiles))
        # Generate a temporary name - eg fsl-<mypid>-date
        tempname = '-'.join( ('fsl', str(getpid()), str(time())) )
        MsgUser.debug("Untarring to %s into temporary folder %s." % (self.location, tempname))
        tempfolder = "/".join((self.location.rstrip('/'),tempname))
        a_result = run_cmd("mkdir %s" % (tempfolder), as_root = as_root)
        if a_result.status == FslIResult.ERROR:
            return a_result
        MsgUser.debug("Calling tar -C %s -x -v -f %s" % (tempfolder, targz_file))
        tar_cmd = ' '.join(('tar',
                            '-C',
                            tempfolder,
                            '-x',
                            '-o',
                            '-f',
                            targz_file))
        #a_result = run_cmd_countlines(tar_cmd, nfiles, tui=False, as_root=as_root)
        a_result = run_cmd_dropstdout(tar_cmd, as_root=as_root)
        try:
            if a_result.status == FslIResult.SUCCESS:
                new_fsl = "/".join((tempfolder.rstrip("/"), 'fsl'))
                install_to = "/".join((self.location.rstrip("/"),"fsl"))
                if path.exists(install_to):
                    # move old one out of way
                    a_result = get_installed_version(install_to)
                    old_version = a_result.result
                    if a_result.status != FslIResult.SUCCESS:
                        MsgUser.warning("The contents of %s doesn't look like an FSL installation! - moving to fsl-0.0.0" % (install_to))
                    old_fsl = '-'.join((install_to, str(old_version)))
                    if path.exists(old_fsl):
                        MsgUser.debug("Looks like there is another copy of the old version of FSL - deleting...")
                        c_result = safe_delete(old_fsl, as_root)
                        if c_result.status == FslIResult.ERROR:
                            raise InstallFailed(";".join(("Install location already has a %s - I've tried to delete it but failed" % (old_fsl), c_result.message)))
                    a_result = run_cmd(" ".join(('mv', install_to, old_fsl)), as_root)
                    if a_result.status == FslIResult.ERROR:
                        # failed to move the old version
                        if not self.keep:
                            c_result = safe_delete(install_to, as_root)
                            if c_result.status == FslIResult.ERROR:
                                raise InstallFailed(";".join((a_result.message, c_result.message)))
                            else:
                                MsgUser.debug("Failed to move old version - %s, but able to delete it." % (a_result.message))
                        else:
                            c_result = safe_delete(tempfolder, as_root)
                            if c_result.status == FslIResult.ERROR:
                                MsgUser.debug("Failed to delete %s - %s." % (tempfolder, c_result.message))
                                raise InstallFailed( ";".join((a_result.message, c_result.message)))
                            else:
                                MsgUser.debug("Deleted temp folder %s - %s." % (tempfolder, c_result.message))
                            MsgUser.debug("Failed to move old version - %s." % (a_result.message))
                            raise InstallFailed('Failed to move old version out of way and you requested it was kept.')

                    # Remove old version if requested
                    if self.keep:
                        MsgUser.message("Old version moved to %s" % (old_fsl))
                    else:
                        c_result = safe_delete(old_fsl, as_root)
                        if c_result.status == FslIResult.ERROR:
                            raise InstallFailed( c_result.message )
                        else:
                            MsgUser.debug("Removed %s" % (old_fsl))  
                MsgUser.debug( " ".join(('mv', new_fsl, install_to)) )
                a_result = run_cmd(" ".join(('mv', new_fsl, install_to)), as_root)
                if a_result.status == FslIResult.ERROR:
                    # Unable to move new install into place
                    MsgUser.debug("Move %s into %s failed - %s." % (new_fsl, install_to, a_result.message))
                    raise InstallFailed('Failed to move new version into place %s' % (a_result.message))

            else:
                MsgUser.debug("Unable to unpack new version - %s." % (a_result.message))
                raise InstallFailed('Unable to unpack new version - %s' % (a_result.message))
        except InstallFailed, e:
            # Clean up unpacked version
            safe_delete(tempfolder,as_root)
            return FslIResult('', FslIResult.ERROR, str(e))

        safe_delete(tempfolder,as_root)
        MsgUser.debug("Install complete")
        MsgUser.ok("FSL software installed.")
        return FslIResult('', FslIResult.SUCCESS, '')
    def install_update(self,mirror):
        import sys
        MsgUser.message("New version %s available. Please run the installer again without the '-c' option to install a fresh copy of the new release." % (str(self.version)))
        sys.exit(0)	
        return FslIResult('', FslIResult.SUCCESS, '')
      
class InstallQuestions(object):
    def __init__(self):
        self.questions = {}
        self.answers = {}
        self.validators = {}
        self.type = {}
    
    def add_question(self, key, question, default, qtype, validation_f):
        self.questions[key] = question
        self.answers[key] = default
        self.type[key] = qtype
        self.validators[key] = validation_f
    
    def ask_question(self, key, tui=False):
        # Ask a question
        no_answer = True
        validator = self.validators[key]

        if self.answers[key] != '':
            default = ''.join( ('[', self.answers[key], ']') )
        else:
            default = ''
        while no_answer:

            if tui:
                tui.info.addstr("%s? %s:" % ( self.questions[key], default))
            else:
                MsgUser.question( "%s? %s:" % ( self.questions[key], default) )

            if tui:
                your_answer = tui.status.getstr()
            else:
                your_answer = raw_input()
                MsgUser.debug("Your answer was %s" % (your_answer))
            if your_answer == '':
                MsgUser.debug("You want the default")
                your_answer = self.answers[key]
            
            if validator(your_answer):
                if self.type[key] == 'bool':
                    if your_answer.lower() == 'yes':
                        self.answers[key] = True
                    else:
                        self.answers[key] = False
                else:
                    self.answers[key] = your_answer
                no_answer = False
        MsgUser.debug("Returning the answer %s" % (self.answers[key]))   
        return self.answers[key]

def yes_no(answer):
    if answer.lower() == 'yes' or answer.lower() == 'no':
        return True
    else:
        MsgUser.message("Please enter yes or no.")
    return False

def check_install_location(folder):
    '''Don't allow relative paths'''
    if folder == '.' or folder == '..' or folder.startswith('./') or folder.startswith('../') or folder.startswith('~'):
        MsgUser.message("Please enter an absolute path.")
        return False
    return True

def external_validate(what_to_check):
    '''We will validate elsewhere'''
    return True

def check_fsl_install(folder):
    '''Check if this folder contains FSL install'''
    from os import path
    fsldir = '/'.join((folder,'fsl'))
    if path.isdir(fsldir):
        return True
    return False


class Tui(object):
    def __init__(self, scr):
        # Build installer screen layout
        # _____________________________
        # |        Title               |
        # -----------------------------
        # Info window
        #
        #
        #
        #
        #
        # _____________________________
        # | progress bar               |
        # -----------------------------
        (height, width) = scr.getmaxyx()
        self.title = curses.newwin(2, width, 0, 0)
        self.progress = curses.newwin(3, width, height - 3, 0)
        self.info = curses.newwin(height - 5, width - 2, 2, 1)

def installer(options, args):
    from os import path,getuid,makedirs, remove
    from sys import argv
    import errno
    upgrade = False
    title="--- FSL Installer - Version %s ---" % (version)
    #init_colours()
    scr=False
    #my_win = Tui(scr)
    #my_win.title.addstr(0,centred_x(scr, title),title, curses.A_REVERSE)
    #scr.refresh()
    MsgUser.message( shell_colours.bold + title + shell_colours.default)

    main_server = 'fsl.fmrib.ox.ac.uk'
    try:
        this_computer = Os()
    except UnsupportedOs, e:
        MsgUser.debug(str(e))
        raise InstallFailed(str(e))

    inst_qus = InstallQuestions()
    inst_qus.add_question('location', "Where would you like to install FSL", '/usr/local', 'path', check_install_location)
    inst_qus.add_question('del_old', "FSL exists in the current location, would you like to keep the old version", 'No', 'bool', yes_no)
    inst_qus.add_question('create', "Install location doesn't exist, should I create it", 'yes', 'bool', yes_no)
    inst_qus.add_question('inst_loc', "Which folder is FSL installed in (without 'fsl')", '/usr/local', 'path', check_fsl_install)
    inst_qus.add_question('skipmd5', 'I was unable to download the checksum of the install file so cannot confirm it is correct. Would you like to install anyway', 'no', 'bool', yes_no)
    inst_qus.add_question('overwrite', "There is already a local copy of the file, would you like to overwrite it", "Yes", 'bool', yes_no)
    my_uid = getuid()
    lv_string = 'latest-version.txt'
    if options.test_installer: lv_string += '-testing'
    if my_uid != 0:
        if options.quiet:
            print 'We may need administrator rights, but you have specified fully automated mode - you may still be asked for an admin password if required.'
            print 'To install fully automatedly, either ensure this is running as the root user (use sudo) or that you can write to the folder you wish to install FSL in.'
        elif not options.download:
            MsgUser.warning("Some operations of the installer require administative rights, for example installing into the default folder of /usr/local. If your account is an 'Administrator' (you have 'sudo' rights) then you will be prompted for your administrator password when necessary.")
    if options.download:
        # Only download the install files then exit
        a_result = fastest_mirror(main_site = '/'.join(('http:/', main_server.rstrip('/'), 'fsldownloads')))
        
        if a_result.status == FslIResult.ERROR:
            # We can't find the FSL site - possibly the internet is down
            MsgUser.failed(a_result.message)
            exit(1)
        
        mirror = a_result.result
        a_result = get_web_version('/'.join((mirror.rstrip('/'), lv_string)))
        if a_result.status == FslIResult.ERROR:
            # We can't find the latest version on the FSL site
            raise InstallFailed(a_result.message)
        else:
            new_version = a_result.result
        
        this_install = FslInstall('/tmp', new_version, mirror, False, which_shell())
        local_file = this_install.url.split('/')[-1]
        if path.exists(local_file):
            if path.isfile(local_file):
                overwrite = inst_qus.ask_question('overwrite')
                if overwrite:
                    MsgUser.warning("Erasing existing file %s" % local_file)
                    try:
                        remove(local_file)
                    except:
                        raise InstallFailed("Unabled to remove local file %s - remove it and try again" % local_file)
                else:
                    raise InstallFailed("Aborting download")
            else:
                raise InstallFailed("There is a directory named %s - cannot overwrite" % local_file)
                
        MsgUser.message("Downloading FSL release to file %s (this may take some time)." % (local_file))
        a_result = download_file(url=this_install.url, localf=local_file, tui=scr)
        if a_result.status == FslIResult.ERROR:
            MsgUser.debug(a_result.message)
            raise InstallFailed(a_result.message)
        MsgUser.message( "Getting MD5 checksum..." )
        md5_url = "/".join((this_install.mirror.rstrip('/'), 'md5sums', this_install.tarf + '.md5'))
        # get the MD5 of the file
        getAndReportMD5(md5_url)
        exit(0)
                    
    if options.env_only and my_uid != 0:
        # No point configuring the root user's environment
        # User has requested environment setup
        fsldir = get_fsldir()
        if fsldir == '':
            if options.d_dir and check_fsl_install(options.d_dir):
                fsldir = '/'.join((options.d_dir.strip('\n').rstrip('/'), 'fsl'))
            else:
                # We can't find an installed version of FSL!
                if not MsgUser.isquiet():
                    fsldir = inst_qus.ask_question('inst_loc') + "/fsl"
                else:
                    raise InstallFailed("I can't locate FSL, try again using '-d <FSLDIR>' to specify where to find the FSL install")

        this_install = FslInstall(fsldir, Version('0.0.0'), '/'.join(('http:/',main_server.rstrip('/'),'fsldownloads')), False, which_shell())
        a_result = this_install.configure_env()
        if a_result.status == FslIResult.ERROR:
            MsgUser.debug(a_result.message)
            raise InstallFailed("Failed to update your profile %s with FSL settings" % (this_install.shell))
        else:
            MsgUser.ok("User profile updated with FSL settings.")
            exit(0)
    if options.tarf:
        # This is a manual install
        if path.isfile(options.tarf):
            if not options.skipmd5:
                # Verify the tar file with the provided MD5 sum
                if options.md5sum:
                    if md5File(options.tarf) != options.md5sum:
                        raise InstallFailed("Install file failed verification.")
                    else:
                        MsgUser.ok("FSL install file verified.")
                else:
                    raise InstallFailed("MD5 sum of installer must be specified for a manual install without MD5 skip enabled")
            fsldir = get_fsldir()
            if fsldir == '':
                if options.d_dir:
                    fsldir = '/'.join((options.d_dir.rstrip('/'), "fsl") )
                else:
                    # We can't find an installed version of FSL!
                    raise InstallFailed("I can't locate existing FSL and you haven't specified an install location.")
            a_result = tarf_version(options.tarf)
            if a_result.status == FslIResult.ERROR:
                raise InstallFailed("Install file doesn't look right")
            this_install = FslInstall(fsldir, a_result.result, '', False, which_shell())
            a_result = this_install.install_tar(options.tarf)
            if a_result.status == FslIResult.ERROR:
                raise InstallFailed("Installation of provided FSL release failed (%s)" % (a_result.message))
        else:
            MsgUser.failed("The specified install file doesn't look like it is a file!")  
            exit(1)
    else:
        a_result = fastest_mirror()
        
        if a_result.status == FslIResult.ERROR:
            # We can't find the FSL site - possibly the internet is down
            MsgUser.failed(a_result.message)
            exit(1)
        
        mirror = a_result.result
        
        if 'fslinstaller' in argv[0]:
            # If this isn't a standalone install check for updates to the installer
            a_result = self_update(mirror)
            if a_result.status == FslIResult.WARN:
                # Installer up-to-date
                MsgUser.skipped(a_result.message)
            elif a_result.status == FslIResult.SUCCESS:
                MsgUser.ok(a_result.message)
        else:
            # We are now running the newly downloaded installer
            MsgUser.ok('Installer updated to latest version %s' % (str(version)))
            fsldir = get_fsldir()
            if fsldir != '':               
                # Install this new installer version into existing FSL install
                # Do we need to be root?
                targetfolder = "/".join((fsldir.rstrip('/'), "etc"))
                as_root = False
                installer = path.abspath(__file__)
                try:
                    if not is_writeable(targetfolder):
                        if not is_writeable_as_root(targetfolder):
                            raise InstallFailed('Target not writeable as super user')
                        else:
                            as_root = True
                except InstallFailed:
                    MsgUser.warning("Installer updated, but cannot be copied to existing FSL install.")
                else:
                    a_result = copy_file(installer, "/".join((targetfolder, "fslinstaller.py")), as_root )
                    if a_result.status == FslIResult.ERROR:
                        MsgUser.warning(a_result.message)
            MsgUser.ok("Installer self update successful.")
        if options.update:
            # Start an update
            MsgUser.message("Looking for new version.")
            fsldir = get_fsldir()
            if fsldir == '':
                if options.d_dir:
                    fsldir = '/'.join((options.d_dir.rstrip('/'), "fsl") )
                else:
                    # We can't find an installed version of FSL!
                    raise InstallFailed("I can't locate FSL, try again using '-d <DESTDIR>' to specify the folder where FSL is installed.")
            a_result = get_installed_version(fsldir)
            if a_result.status == FslIResult.ERROR:
                # We can't find an installed version of FSL!
                raise InstallFailed(a_result.message)
            else:
                this_version = a_result.result
                MsgUser.debug("You have version %s" % (this_version))
               
                a_result = get_web_version('/'.join((mirror.rstrip('/'), lv_string)))
                if a_result.status == FslIResult.ERROR:
                    # We can't find the latest version on the FSL site
                    raise InstallFailed(a_result.message)
                else:
                    new_version = a_result.result
                    if new_version > this_version:
                        # Update Available
                        if new_version.major > this_version.major:
                            # We don't support patching between major versions so download a fresh copy
                            upgrade=True
                        else:
                            # Get the update and call the patcher
                            this_install = FslInstall(fsldir, new_version, mirror, False, which_shell())
                            a_result = this_install.install_update(mirror)
                            if a_result.status == FslIResult.ERROR:
                                # Updated files
                                raise InstallFailed(a_result.message)
                            else:
                                MsgUser.ok('FSL updated to %s' % (new_version))
                    else:
                        MsgUser.ok('You have the latest version.')
                        return
            if not upgrade: return
        keep=False    
        if not upgrade:
            if not options.d_dir:
                instdir = inst_qus.ask_question('location')
            else:
                instdir = options.d_dir
                MsgUser.message("Attempting to install to %s" % (instdir))
            fsldir = '/'.join( (instdir.rstrip('/'), "fsl") )
            MsgUser.debug("Install folder %s, FSLDIR is %s" % (instdir, fsldir))
            if path.isdir(fsldir) and not options.quiet:
                keep = inst_qus.ask_question('del_old')
            else:
                keep = False
                if not path.isdir(instdir):
                    if inst_qus.ask_question('create'):
                        # Make the install directory after checking if we can write here
                        try:
                            MsgUser.debug('Creating folder %s' % (instdir))
                            makedirs(instdir)
                        except OSError, e:
                            if e.errno == errno.EACCES:
                                a_result = run_cmd("mkdir -p %s" % (instdir), as_root=True)
                                if a_result.status == FslIResult.ERROR:
                                    # Cannot create folder abort
                                    MsgUser.failed('Failed to create folder (%s). Aborting install.' % (a_result.message))
                                    return
                            elif e.errno != errno.EEXIST:
                                raise
                    else:
                        MsgUser.failed('Aborting install.')
                        return
            # Do the install
            a_result = get_web_version('/'.join((mirror.rstrip('/'), lv_string)))
            if a_result.status == FslIResult.ERROR:
                # We can't find the latest version on the FSL site
                raise InstallFailed(a_result.message)
            else:
                new_version = a_result.result

        this_install = FslInstall(fsldir, new_version, mirror, keep, which_shell())
        (_, local_fname) = tempFileName(close=True)
        MsgUser.message("Downloading FSL version %s (this may take some time)" % (new_version))
        a_result = download_file(url=this_install.url, localf=local_fname, tui=scr)
        if a_result.status == FslIResult.ERROR:
            MsgUser.debug(a_result.message)
            raise InstallFailed(a_result.message)
        if not options.skipmd5:
            MsgUser.message( "Checking download..." )
            md5_url = "/".join((this_install.mirror.rstrip('/'), 'md5sums', this_install.tarf + '.md5'))
            # get the MD5 of the file
            MsgUser.debug("Downloading %s to check %s" % (md5_url,local_fname))
            m_result = getAndVerifyMD5(md5_url, local_fname)
            if m_result.status == FslIResult.SUCCESS:
                MsgUser.debug("MD5 sum of install file %s correct." % (this_install.tarf))
                MsgUser.ok("Download is good.")
            elif m_result.status == FslIResult.WARN:
                if not MsgUser.isquiet():
                    cont = inst_qus.ask_question('skipmd5')
                    if not cont:
                        safe_delete(local_fname)
                        return FslIResult(False, FslIResult.ERROR, '')
                else:
                    safe_delete(local_fname)
                    return FslIResult(False, FslIResult.ERROR, 'Found update to installer but unable to check it is valid. Please try again.')
            else:
                return m_result
        
        a_result = this_install.install_tar(local_fname)
        c_result = safe_delete(local_fname)
        if a_result.status == FslIResult.ERROR or c_result.status == FslIResult.ERROR:
            MsgUser.debug("%s ; %s" % (a_result.message, c_result.message))
            raise InstallFailed("%s ; %s" % (a_result.message, c_result.message))
        # Install myself into the new FSL install
        a_result = this_install.install_installer()
        if a_result.status == FslIResult.ERROR:
            MsgUser.warning("Unable to install this installer into FSL - you will still be able to use FSL.")
        if this_computer.vendor == 'apple':
            results = this_install.slash_applications_links()
            for app in results.values():
                if app.status == FslIResult.WARN:
                    if app.result == False:
                        MsgUser.warning(app.message)
                    else:
                        MsgUser.skipped(app.message)
            a_result = find_X11()
            if a_result.status == FslIResult.WARN:
                MsgUser.warning(a_result.message)
            elif a_result.status == FslIResult.SUCCESS:
                MsgUser.ok(a_result.message)
                
    if not options.skip_env:
        if options.env_all:
            if this_install.computer.os == "linux":
                # Setup the system-wise environment
                a_result = this_install.patch_systemprofile()
                if a_result.status == FslIResult.ERROR:
                    MsgUser.debug(a_result.message)
                    MsgUser.failed("Failed to configure system-wide profile %s with FSL settings" % (this_install.shell))
                else:
                    MsgUser.debug("System-wide profiles setup.")
                    MsgUser.ok("System-wide FSL configuration complete.")
            else:
                MsgUser.skipped("System-wide profiles not supported on %s" % (this_install.computer.os))
        elif my_uid !=0:
            # Setup the environment for the current user
            a_result = this_install.configure_env()
            if a_result.status == FslIResult.ERROR:
                MsgUser.debug(a_result.message)
                MsgUser.failed("Failed to update your profile %s with FSL settings" % (this_install.shell))
            else:
                MsgUser.ok("User profile updated with FSL settings, you will need to log out and back in to use the FSL tools.")
                MsgUser.debug("User's environment (%s) updated." % (this_install.shell))
        
if __name__ == '__main__':

    usage = "usage: %prog [options]"
    ver = "%%prog %s" % (version)
    parser = OptionParser(usage=usage, version=ver)
    parser.add_option("-d", "--dest", dest="d_dir",
                      help="Install into folder given by DESTDIR",
                      metavar="DESTDIR", action="store",
                      type="string")
    parser.add_option("-e", dest="env_only",
                      help="Only setup/update your environment",
                      action="store_true")
    parser.add_option("-E", dest="env_all",
                      help="Setup/update the environment for ALL users",
                      action="store_true")
    parser.add_option("-v", help="Print version number and exit",
                      action="version")
    parser.add_option("-c", "--checkupdate", dest='update',
                      help="Check for FSL updates - needs an internet connection",
                      action="store_true")
    parser.add_option("-o", "--downloadonly", dest="download",
                      help="Only download the FSL install file/MD5 checksum for installation elsewhere without an internet connection. Will store the file in the current folder and print it's expected checksum (to be given to -C option at install time) to the terminal.",
                      action="store_true")

    manual_group = OptionGroup(parser, "Manual Install Options",
                               "These are for use if you have manually downloaded the FSL install .tar.gz file (for example with the -o option).")
    manual_group.add_option("-f", "--file", dest="tarf",
                      help="Install FSL distribution given in TARGZFILE. If you don't specify an install location (with -d) then it will try to find an existing install to replace.",
                      metavar="TARGZFILE", action="store",
                      type="string")
    manual_group.add_option("-q", "--quiet", dest='quiet',
                            help="Silence all messages - useful if scripting install",
                            action="store_true")
    manual_group.add_option("-C", "--md5", dest="md5sum",
                            help="Verify the FSL tar file with MD5 sum given by MD5SUM",
                            metavar="MD5SUM", action="store",
                            type="string")
    manual_group.add_option("-M", "--nomd5", dest="skipmd5",
                            help="Do not verify the FSL tar file",
                            action="store_true")
    manual_group.add_option("-p", dest="skip_env",
                            help="Don't setup the environment",
                            action="store_true")
    parser.add_option_group(manual_group)
    
    debug_group = OptionGroup(parser, "Debugging Options",
                                      "These are for use if you have a problem running this installer.")
    debug_group.add_option("-D", dest="verbose",
                           help="Switch on debug messages",
                           action="store_true")
    debug_group.add_option("-G", dest="test_installer",
                           help=SUPPRESS_HELP,
                           action="store_true")
    parser.add_option_group(debug_group)
    
    (options, args) = parser.parse_args()
    if options.verbose: MsgUser.debugOn()
    if options.quiet: MsgUser.quietOn()
    try:
        #curses.wrapper(installer, options, args)
        installer(options,args)
    except InstallFailed, e:
        MsgUser.failed(e.value)
        exit(1)
    except UnsupportedOs, e:
        MsgUser.failed(e.value)
        exit(1)
    except KeyboardInterrupt, e:
        MsgUser.message('')
        MsgUser.failed("Install aborted.")
        exit(1)

